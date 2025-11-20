# eval_by_index.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

LABELS = "labels_sorted.csv"
PREDS  = "pred_ensemble_sorted.csv"

# 1) 로드
labels = pd.read_csv(LABELS)
preds  = pd.read_csv(PREDS)

# 2) 필요한 컬럼 존재 확인 및 정리
if "label" not in labels.columns:
    raise ValueError("labels_sorted.csv에 'label' 컬럼이 필요합니다.")
if "prediction" not in preds.columns and "score_mean" not in preds.columns:
    raise ValueError("pred_ensemble_sorted.csv에 'prediction' 또는 'score_mean' 컬럼이 필요합니다.")

# prediction이 없으면 score_mean을 0.5 임계치로 이산화
if "prediction" not in preds.columns:
    preds["prediction"] = (pd.to_numeric(preds["score_mean"], errors="coerce") >= 0.5).astype(int)

# 3) 길이 맞추기(순서 기준 1:1 비교)
n_labels = len(labels)
n_preds  = len(preds)
n = min(n_labels, n_preds)

labels_use = labels.iloc[:n].copy().reset_index(drop=True)
preds_use  = preds.iloc[:n].copy().reset_index(drop=True)

if n_labels != n_preds:
    print(f"⚠️ 길이가 다릅니다. labels={n_labels}, preds={n_preds} → 공통 {n}개만 비교합니다.")

# 4) 문자열/숫자 혼용 안전 변환(benign=0, malicious=1)
def to01(s):
    s = s.astype(str).str.strip().str.lower()
    mapping = {"benign":"0", "malicious":"1", "false":"0", "true":"1"}
    s = s.replace(mapping)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

y_true = to01(labels_use["label"])
y_pred = to01(preds_use["prediction"])

# 5) 지표 계산
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred, labels=[0,1])

print(f"✅ 순서(index) 기준 1:1 비교 완료 (n={n})")
print(f"🎯 Accuracy : {acc*100:.2f}%")
print(f"📈 Precision: {prec:.4f}")
print(f"📊 Recall   : {rec:.4f}")
print(f"🏁 F1-score : {f1:.4f}")
print("\n🧩 Confusion Matrix [rows=true 0/1, cols=pred 0/1]:")
print(cm)
print("\n📋 Classification report:")
print(classification_report(y_true, y_pred, target_names=["benign(0)","malicious(1)"], zero_division=0))

# 6) 결과 저장
out = pd.DataFrame({
    "idx": range(n),
    "label": y_true,
    "prediction": y_pred
})
out["correct"] = (out["label"] == out["prediction"])
out.to_csv("merged_eval_by_index.csv", index=False)
out[~out["correct"]].to_csv("wrong_predictions_by_index.csv", index=False)

print("\n💾 저장 완료:")
print("- merged_eval_by_index.csv (비교에 사용된 n개 전부)")
print("- wrong_predictions_by_index.csv (오답만)")
if n_labels != n_preds:
    print(f"- 참고: 비교에 사용되지 않은 남은 행 → labels: {n_labels-n}, preds: {n_preds-n}")
