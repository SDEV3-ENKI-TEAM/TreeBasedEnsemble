import argparse, os, json, joblib, numpy as np, pandas as pd
from trace_features import TraceFeaturizer

def build_matrix(df: pd.DataFrame, feature_columns):
    feats = df.reindex(columns=feature_columns, fill_value=0.0)
    feats = feats.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats.values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Dir with *.json traces to score")
    ap.add_argument("--model_dir", required=True, help="Directory containing ensemble_models.joblib & feature_columns.json")
    ap.add_argument("--out_csv", default="pred_ensemble.csv")
    ap.add_argument("--thr", type=float, default=0.7, help="Threshold for malicious classification")
    args = ap.parse_args()

    folds = joblib.load(os.path.join(args.model_dir, "ensemble_models.joblib"))
    with open(os.path.join(args.model_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    tf = TraceFeaturizer()
    feats = tf.featurize_dir(args.data_dir)

    X = build_matrix(feats, feature_columns)

    p_l_all, p_x_all, p_c_all = [], [], []
    for i, fd in enumerate(folds, 1):
        lgbm = fd["lgbm"]; xgb = fd["xgb"]; cat = fd["cat"]
        p_l_all.append(lgbm.predict_proba(X)[:,1])
        p_x_all.append(xgb.predict_proba(X)[:,1])
        p_c_all.append(cat.predict_proba(X)[:,1])

    p_l = np.mean(p_l_all, axis=0)
    p_x = np.mean(p_x_all, axis=0)
    p_c = np.mean(p_c_all, axis=0)
    p_m = (p_l + p_x + p_c) / 3.0

    out = feats[["traceID"]].copy()
    out["score_lgbm"] = p_l
    out["score_xgb"]  = p_x
    out["score_cat"]  = p_c
    out["score_mean"] = p_m

    # threshold 적용 → 악성/정상 라벨 추가
    out["prediction"] = np.where(out["score_mean"] >= args.thr, "malicious", "benign")

    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[+] Saved {args.out_csv} with {len(out)} rows")
    
    # 콘솔에 요약 출력
    for _, row in out.iterrows():
        print(f"Trace {row['traceID']} → score={row['score_mean']:.4f}, result={row['prediction']}")

if __name__ == "__main__":
    main()
