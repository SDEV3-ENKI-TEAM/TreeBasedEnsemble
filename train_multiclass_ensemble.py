# -*- coding: utf-8 -*-
import os, json, argparse, joblib, warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, log_loss, accuracy_score, roc_auc_score,
    precision_score, recall_score, average_precision_score
)

from trace_features import TraceFeaturizer

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

def build_matrix(df: pd.DataFrame, feature_columns=None):
    feats = df.drop(columns=[c for c in ("traceID","label","behavior_label") if c in df.columns], errors="ignore")
    feats = feats.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_columns is None:
        X = feats.values.astype(np.float32)
        cols = list(feats.columns)
    else:
        X = feats.reindex(columns=feature_columns, fill_value=0.0).values.astype(np.float32)
        cols = list(feature_columns)
    return X, cols

def pr_auc_macro(y_true: np.ndarray, prob: np.ndarray, n_classes: int) -> float:
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return average_precision_score(y_bin, prob, average="macro")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="학습용 *.json 트레이스 폴더")
    ap.add_argument("--labels_csv", required=True, help="traceID,behavior_label CSV (문자열 라벨)")
    ap.add_argument("--out_dir", default="model_mc_out", help="모델 출력 폴더")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 피처 추출
    tf = TraceFeaturizer()
    df = tf.featurize_dir(args.data_dir)

    # 2) 라벨 로드 & 병합
    labels = pd.read_csv(args.labels_csv, encoding="utf-8-sig")
    if not {"traceID","behavior_label"}.issubset(labels.columns):
        raise ValueError("labels_csv는 traceID,behavior_label 컬럼을 포함해야 합니다.")
    df = df.merge(labels[["traceID","behavior_label"]], on="traceID", how="inner")
    df = df.dropna(subset=["behavior_label"])
    if len(df) == 0:
        raise ValueError("라벨과 매칭되는 학습 샘플이 없습니다.")

    # 3) 행렬/라벨 준비
    X, feature_columns = build_matrix(df)
    le = LabelEncoder()
    y = le.fit_transform(df["behavior_label"].astype(str).values)
    classes = list(le.classes_)
    n_classes = len(classes)
    print(f"[INFO] n_samples={len(df)}, n_features={X.shape[1]}, n_classes={n_classes}")
    print(f"[INFO] classes={classes}")

    # 4) 교차검증 설정
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    folds = []
    oof_pred_lgbm = np.zeros((len(df), n_classes), dtype=float)
    oof_pred_xgb  = np.zeros((len(df), n_classes), dtype=float)
    oof_pred_cat  = np.zeros((len(df), n_classes), dtype=float)

    for i, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 5) LightGBM (multiclass)
        lgbm = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=n_classes,
            reg_lambda=1.0,
            random_state=args.seed + i,
            verbosity=-1
        )
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )
        p_l = lgbm.predict_proba(X_va)
        oof_pred_lgbm[va_idx] = p_l

        # 6) XGBoost (multiclass)
        xgb = XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            random_state=args.seed + i,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        p_x = xgb.predict_proba(X_va)
        oof_pred_xgb[va_idx] = p_x

        # 7) CatBoost (multiclass)
        cat = CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            random_seed=args.seed + i,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.8,
            od_type="Iter",
            od_wait=50,
            verbose=False
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
        p_c = cat.predict_proba(X_va)
        oof_pred_cat[va_idx] = p_c

        # ── 폴드별 성능 함수 ──────────────────────────────────────────────
        def fold_metrics(y_true, prob):
            pred = prob.argmax(axis=1)
            f1m  = f1_score(y_true, pred, average="macro")
            prec = precision_score(y_true, pred, average="macro", zero_division=0)
            rec  = recall_score(y_true, pred, average="macro", zero_division=0)
            acc  = accuracy_score(y_true, pred)
            ll   = log_loss(y_true, prob, labels=list(range(n_classes)))
            try:
                auc = roc_auc_score(y_true, prob, multi_class="ovr")
            except Exception:
                auc = np.nan
            try:
                pr_auc = pr_auc_macro(y_true, prob, n_classes)
            except Exception:
                pr_auc = np.nan
            return f1m, prec, rec, acc, ll, auc, pr_auc

        f1_l, pre_l, rec_l, acc_l, ll_l, auc_l, pra_l = fold_metrics(y_va, p_l)
        f1_x, pre_x, rec_x, acc_x, ll_x, auc_x, pra_x = fold_metrics(y_va, p_x)
        f1_c, pre_c, rec_c, acc_c, ll_c, auc_c, pra_c = fold_metrics(y_va, p_c)
        p_m = (p_l + p_x + p_c) / 3.0
        f1_m, pre_m, rec_m, acc_m, ll_m, auc_m, pra_m = fold_metrics(y_va, p_m)

        print(
            f"[Fold {i}] "
            f"LGBM ROC-AUC={auc_l:.4f} PR-AUC={pra_l:.4f} | Precision/Recall/F1={pre_l:.3f}/{rec_l:.3f}/{f1_l:.3f} | acc={acc_l:.4f} ll={ll_l:.4f} || "
            f"XGB ROC-AUC={auc_x:.4f} PR-AUC={pra_x:.4f} | Precision/Recall/F1={pre_x:.3f}/{rec_x:.3f}/{f1_x:.3f} | acc={acc_x:.4f} ll={ll_x:.4f} || "
            f"CAT ROC-AUC={auc_c:.4f} PR-AUC={pra_c:.4f} | Precision/Recall/F1={pre_c:.3f}/{rec_c:.3f}/{f1_c:.3f} | acc={acc_c:.4f} ll={ll_c:.4f} || "
            f"MEAN ROC-AUC={auc_m:.4f} PR-AUC={pra_m:.4f} | Precision/Recall/F1={pre_m:.3f}/{rec_m:.3f}/{f1_m:.3f} | acc={acc_m:.4f} ll={ll_m:.4f}"
        )

        folds.append({"lgbm": lgbm, "xgb": xgb, "cat": cat})

    # 8) OOF 요약
    def oof_metrics(y_true, prob):
        pred = prob.argmax(axis=1)
        f1m  = f1_score(y_true, pred, average="macro")
        prec = precision_score(y_true, pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, pred, average="macro", zero_division=0)
        acc  = accuracy_score(y_true, pred)
        ll   = log_loss(y_true, prob, labels=list(range(n_classes)))
        try:
            auc = roc_auc_score(y_true, prob, multi_class="ovr")
        except Exception:
            auc = np.nan
        try:
            pr_auc = pr_auc_macro(y_true, prob, n_classes)
        except Exception:
            pr_auc = np.nan
        return f1m, prec, rec, acc, ll, auc, pr_auc

    f1_l, pre_l, rec_l, acc_l, ll_l, auc_l, pra_l = oof_metrics(y, oof_pred_lgbm)
    f1_x, pre_x, rec_x, acc_x, ll_x, auc_x, pra_x = oof_metrics(y, oof_pred_xgb)
    f1_c, pre_c, rec_c, acc_c, ll_c, auc_c, pra_c = oof_metrics(y, oof_pred_cat)
    oof_mean = (oof_pred_lgbm + oof_pred_xgb + oof_pred_cat) / 3.0
    f1_m, pre_m, rec_m, acc_m, ll_m, auc_m, pra_m = oof_metrics(y, oof_mean)

    print(f"[OOF] LGBM ROC-AUC={auc_l:.4f} PR-AUC={pra_l:.4f} | "
      f"Precision/Recall/F1={pre_l:.3f}/{rec_l:.3f}/{f1_l:.3f}")

    print(f"[OOF] XGB  ROC-AUC={auc_x:.4f} PR-AUC={pra_x:.4f} | "
        f"Precision/Recall/F1={pre_x:.3f}/{rec_x:.3f}/{f1_x:.3f}")

    print(f"[OOF] CAT  ROC-AUC={auc_c:.4f} PR-AUC={pra_c:.4f} | "
        f"Precision/Recall/F1={pre_c:.3f}/{rec_c:.3f}/{f1_c:.3f}")

    print(f"[OOF] MEAN ROC-AUC={auc_m:.4f} PR-AUC={pra_m:.4f} | "
        f"Precision/Recall/F1={pre_m:.3f}/{rec_m:.3f}/{f1_m:.3f}")

    # 9) 저장
    joblib.dump(folds, os.path.join(args.out_dir, "ensemble_models_mc.joblib"))
    with open(os.path.join(args.out_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

    meta = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "classes": classes,
        "oof": {
            "LGBM": {"f1": f1_l, "precision": pre_l, "recall": rec_l, "acc": acc_l, "logloss": ll_l, "roc_auc_ovr": auc_l, "pr_auc_macro": pra_l},
            "XGB":  {"f1": f1_x, "precision": pre_x, "recall": rec_x, "acc": acc_x, "logloss": ll_x, "roc_auc_ovr": auc_x, "pr_auc_macro": pra_x},
            "CAT":  {"f1": f1_c, "precision": pre_c, "recall": rec_c, "acc": acc_c, "logloss": ll_c, "roc_auc_ovr": auc_c, "pr_auc_macro": pra_c},
            "MEAN": {"f1": f1_m, "precision": pre_m, "recall": rec_m, "acc": acc_m, "logloss": ll_m, "roc_auc_ovr": auc_m, "pr_auc_macro": pra_m},
        }
    }
    with open(os.path.join(args.out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[+] Saved models to: {args.out_dir}")
    print(f"    - ensemble_models_mc.joblib, feature_columns.json, label_mapping.json, train_meta.json")

if __name__ == "__main__":
    main()
