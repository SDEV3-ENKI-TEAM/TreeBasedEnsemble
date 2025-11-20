#!/usr/bin/env python3
import json, os, glob, argparse, pandas as pd

def safe_load_json(path):
    # UTF-8 BOM 대응
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
    try:
        return json.loads(text)
    except Exception:
        with open(path, "r", encoding="utf-8") as f2:
            return json.loads(f2.read().strip())

def tags_to_dict(tags):
    out = {}
    for t in tags or []:
        k = t.get("key")
        v = t.get("value")
        if k is not None:
            out[k] = v
    return out

def extract_trace_metrics(trace_json):
    spans = trace_json.get("spans", [])
    span_count = len(spans)
    total_rule_matches = 0
    for sp in spans:
        tdict = tags_to_dict(sp.get("tags", []))
        mc = tdict.get("sigma.match_count")
        if isinstance(mc, int):
            total_rule_matches += mc
        else:
            desc = tdict.get("otel.status_description")
            if isinstance(desc, str) and "Sigma rules matched" in desc:
                try:
                    total_rule_matches += int(desc.split(":")[-1].strip())
                except Exception:
                    pass
    return span_count, total_rule_matches

def main():
    ap = argparse.ArgumentParser(description="폴더 내 트레이스 파일들의 평균 스팬/룰매칭 계산기")
    ap.add_argument("-d", "--dir", default="./traces_train/malware", help="트레이스 JSON들이 있는 폴더 (기본: 현재 폴더)")
    ap.add_argument("-o", "--out", default="trace_metrics_summary.csv", help="CSV 출력 경로")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    rows = []
    for path in files:
        try:
            data = safe_load_json(path)
            span_count, rule_matches = extract_trace_metrics(data)
            rows.append({
                "file": os.path.basename(path),
                "span_count": span_count,
                "rule_matches_sum": rule_matches,
                "rule_matches_per_span": (rule_matches / span_count) if span_count else None,
            })
        except Exception as e:
            rows.append({
                "file": os.path.basename(path),
                "span_count": None,
                "rule_matches_sum": None,
                "rule_matches_per_span": None,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        valid = df.dropna(subset=["span_count", "rule_matches_sum"])
        avg_span = valid["span_count"].mean() if not valid.empty else None
        avg_rules = valid["rule_matches_sum"].mean() if not valid.empty else None
        avg_rules_per_span = valid["rule_matches_per_span"].mean() if not valid.empty else None

        print("=== Per-file metrics ===")
        print(df.to_string(index=False))

        print("\n=== Overall averages (across traces) ===")
        print(f"files_counted: {int(valid.shape[0]) if not valid.empty else 0}")
        print(f"avg_span_count_per_trace: {avg_span}")
        print(f"avg_rule_matches_per_trace: {avg_rules}")
        print(f"avg_rule_matches_per_span: {avg_rules_per_span}")

        df.to_csv(args.out, index=False)
        print(f"\nSaved CSV -> {args.out}")
    else:
        print("해당 폴더에 *.json 파일이 없습니다.")

if __name__ == "__main__":
    main()
