# -*- coding: utf-8 -*-
import os, json, argparse, joblib
import numpy as np
import pandas as pd

from trace_features import TraceFeaturizer

def build_matrix(df: pd.DataFrame, feature_columns):
    feats = df.drop(columns=[c for c in ("traceID","label","behavior_label") if c in df.columns], errors="ignore")
    feats = feats.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = feats.reindex(columns=feature_columns, fill_value=0.0).values.astype(np.float32)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="예측용 *.json 트레이스 폴더")
    ap.add_argument("--model_dir", required=True, help="train_multiclass_ensemble.py 산출 폴더")
    ap.add_argument("--out_csv", default="pred_multiclass.csv")
    args = ap.parse_args()

    folds = joblib.load(os.path.join(args.model_dir, "ensemble_models_mc.joblib"))
    feature_columns = json.load(open(os.path.join(args.model_dir, "feature_columns.json"), "r", encoding="utf-8"))
    label_mapping = json.load(open(os.path.join(args.model_dir, "label_mapping.json"), "r", encoding="utf-8"))
    classes = label_mapping["classes"]
    n_classes = len(classes)

    tf = TraceFeaturizer()
    feats = tf.featurize_dir(args.data_dir)
    X = build_matrix(feats, feature_columns)

    # 모델별·폴드별 확률 평균
    P = []
    for fd in folds:
        p_l = fd["lgbm"].predict_proba(X)
        p_x = fd["xgb"].predict_proba(X)
        p_c = fd["cat"].predict_proba(X)
        P.append((p_l + p_x + p_c) / 3.0)
    proba = np.mean(P, axis=0)  # (N, n_classes)
    pred_idx = proba.argmax(axis=1)
    pred_lbl = [classes[i] for i in pred_idx]

    # 출력
    out = pd.DataFrame({
        "traceID": feats["traceID"].values,
        "predicted_behavior": pred_lbl
    })
    # 각 클래스 확률도 컬럼으로 추가
    for j, cls in enumerate(classes):
        out[f"prob_{cls}"] = proba[:, j]

    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[+] Saved: {args.out_csv}")
    print(f"    n_samples={len(out)}, n_classes={n_classes}, classes={classes}")

if __name__ == "__main__":
    main()
