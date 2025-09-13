import argparse, os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from trace_features import TraceFeaturizer

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier


def build_matrix(df: pd.DataFrame, feature_columns=None):
    drop_cols = {"traceID", "label"}
    if feature_columns is not None:
        X = df.reindex(columns=feature_columns, fill_value=0.0)
        return X.values, list(X.columns)
    feats = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    feats = feats.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats.values, list(feats.columns)

def bin_metrics(y_true, y_prob, thr=0.6):
    """임계값 기반 이진 분류 지표 (Precision/Recall/F1)"""
    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with *.json traces")
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: traceID,label (1=malicious,0=benign)")
    ap.add_argument("--out_dir", default="model_ens_out", help="Output directory")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr", type=float, default=0.6, help="Precision/Recall/F1 계산용 임계값")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load features
    tf = TraceFeaturizer()
    feats = tf.featurize_dir(args.data_dir)
    labels = pd.read_csv(args.labels_csv)
    df = pd.merge(feats, labels, on="traceID", how="inner")
    assert "label" in df.columns, "labels_csv must have column 'label'"

    X, feature_columns = build_matrix(df)
    y = df["label"].astype(int).values

    with open(os.path.join(args.out_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_columns, f)

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = float(neg / max(1, pos)) if pos > 0 else 1.0

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    oof_mean = np.zeros(len(y), dtype=float)
    oof_lgbm = np.zeros(len(y), dtype=float)
    oof_xgb  = np.zeros(len(y), dtype=float)
    oof_cat  = np.zeros(len(y), dtype=float)

    folds = []
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        # LightGBM
        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            reg_lambda=1.0,
            scale_pos_weight=max(1.0, scale_pos_weight),
            random_state=args.seed + i,
            verbosity=-1
        )
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="logloss",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)
            ]
        )

        #XGBoost
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            gamma=0.0,
            objective="binary:logistic",
            tree_method="hist",
            random_state=args.seed + i,
            scale_pos_weight=max(1.0, scale_pos_weight),
            early_stopping_rounds=50,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            enable_categorical=False
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        # CatBoost
        cat = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            random_seed=args.seed + i,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.8,
            class_weights=[1.0, max(1.0, scale_pos_weight)],
            verbose=False,
            od_type="Iter",
            od_wait=50
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)

        p_l = lgbm.predict_proba(X_va)[:, 1]
        p_x = xgb.predict_proba(X_va)[:, 1]
        p_c = cat.predict_proba(X_va)[:, 1]
        p_m = (p_l + p_x + p_c) / 3.0

        oof_lgbm[va] = p_l
        oof_xgb[va]  = p_x
        oof_cat[va]  = p_c
        oof_mean[va] = p_m

        folds.append({"lgbm": lgbm, "xgb": xgb, "cat": cat})


    # ── OOF 지표 계산 및 출력 ─────────────────────────────────────────────
    auc_l = roc_auc_score(y, oof_lgbm); ap_l = average_precision_score(y, oof_lgbm)
    auc_x = roc_auc_score(y, oof_xgb ); ap_x = average_precision_score(y, oof_xgb )
    auc_c = roc_auc_score(y, oof_cat ); ap_c = average_precision_score(y, oof_cat )
    auc_m = roc_auc_score(y, oof_mean); ap_m = average_precision_score(y, oof_mean)

    pre_l, rc_l, f1_l = bin_metrics(y, oof_lgbm, thr=args.thr)
    pre_x, rc_x, f1_x = bin_metrics(y, oof_xgb,  thr=args.thr)
    pre_c, rc_c, f1_c = bin_metrics(y, oof_cat,  thr=args.thr)
    pre_m, rc_m, f1_m = bin_metrics(y, oof_mean, thr=args.thr)

    print(f"[OOF] LGBM ROC-AUC={auc_l:.4f} PR-AUC={ap_l:.4f} | "
      f"Precision/Recall/F1={pre_l:.3f}/{rc_l:.3f}/{f1_l:.3f}")

    print(f"[OOF] XGB  ROC-AUC={auc_x:.4f} PR-AUC={ap_x:.4f} | "
        f"Precision/Recall/F1={pre_x:.3f}/{rc_x:.3f}/{f1_x:.3f}")

    print(f"[OOF] CAT  ROC-AUC={auc_c:.4f} PR-AUC={ap_c:.4f} | "
        f"Precision/Recall/F1={pre_c:.3f}/{rc_c:.3f}/{f1_c:.3f}")

    print(f"[OOF] MEAN ROC-AUC={auc_m:.4f} PR-AUC={ap_m:.4f} | "
        f"Precision/Recall/F1={pre_m:.3f}/{rc_m:.3f}/{f1_m:.3f}")
    
    # ── 저장 ──────────────────────────────────────────────────────────────
    joblib.dump(folds, os.path.join(args.out_dir, "ensemble_models.joblib"))
    print(f"[+] Saved {len(folds)}-fold ensemble to {os.path.join(args.out_dir, 'ensemble_models.joblib')}")
    print(f"[+] Feature columns saved to feature_columns.json")

if __name__ == "__main__":
    main()
