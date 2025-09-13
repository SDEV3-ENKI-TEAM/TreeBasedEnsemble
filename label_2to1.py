import pandas as pd

# --------------------------
# 1) 이진 분류용 labels.csv
# --------------------------
ben = pd.read_csv("labels_benign.csv")   # 정상 → 0
mal = pd.read_csv("labels_mal.csv")      # 악성 → 1
labels = pd.concat([ben, mal], ignore_index=True).drop_duplicates("traceID")
labels.to_csv("labels.csv", index=False, encoding="utf-8")
print(f"[+] Saved labels.csv (n={len(labels)})")

# --------------------------
# 2) 행위 분류용 behaviors.csv
# --------------------------
# ransomware / trojan / 기타 라벨 CSV들을 모두 합치세요
ransom = pd.read_csv("labels_ransomware.csv")  # 라벨= "ransomware"
trojan = pd.read_csv("labels_troijan.csv")      # 라벨= "trojan"
infostealer = pd.read_csv("labels_infostealer.csv")
RAT = pd.read_csv("labels_RAT.csv")
worm = pd.read_csv("labels_worm.csv")

behaviors = pd.concat([ransom, trojan, infostealer, RAT, worm], ignore_index=True).drop_duplicates("traceID")
behaviors.rename(columns={"label": "behavior_label"}, inplace=True)
behaviors.to_csv("behaviors.csv", index=False, encoding="utf-8")
print(f"[+] Saved behaviors.csv (n={len(behaviors)})")
