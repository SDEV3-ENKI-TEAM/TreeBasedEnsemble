"""
한 폴더 내의 모든 *.json 트레이스에 동일 라벨을 부여해 labels.csv 생성
- 라벨은 정수/문자열 모두 허용
사용 예:
    # 이진 분류용 (정수 라벨)
    python make_labels_from_folder.py --data_dir traces_train/benign --label 0 --out_csv labels_benign.csv
    python make_labels_from_folder.py --data_dir traces_train/malware --label 1 --out_csv labels_mal.csv

    # 행위 분류용 (문자열 라벨)
    python make_labels_from_folder.py --data_dir traces/ransomware --label ransomware --out_csv labels_ransomware.csv
"""
import argparse, os, json, pandas as pd

def iter_json_files(path):
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(".json"):
                yield os.path.join(root, fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="*.json 트레이스 폴더")
    ap.add_argument("--label", required=True, help="부여할 라벨 (정수 또는 문자열)")
    ap.add_argument("--out_csv", required=True, help="출력 CSV 경로")
    args = ap.parse_args()

    # 라벨은 문자열로 저장, 숫자도 str("0") → 나중에 필요시 int 변환 가능
    label_value = args.label

    rows = []
    for fp in iter_json_files(args.data_dir):
        try:
            with open(fp, "r", encoding="utf-8-sig") as f:  # BOM 안전
                j = json.load(f)
            tid = j.get("traceID") or j.get("traceId") or j.get("trace_id")
            if not tid and isinstance(j, dict) and "spans" in j and j["spans"]:
                tid = j["spans"][0].get("traceID")
            if not tid:
                print(f"[!] traceID 미발견: {fp}")
                continue
            rows.append({"traceID": tid, "label": label_value})
        except Exception as e:
            print(f"[!] 읽기 실패: {fp} ({e})")

    pd.DataFrame(rows).drop_duplicates(subset=["traceID"]).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[+] Saved {args.out_csv}  (n={len(rows)})")

if __name__ == "__main__":
    main()
