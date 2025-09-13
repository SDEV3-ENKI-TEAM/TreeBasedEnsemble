# 트리 기반 앙상블 모델

트레이스 데이터를 이용하여 이진 분류(정상/악성)와 다중 분류(행위 유형 분류)를 수행하는 앙상블 모델. 정형 피처를 활용한 부스팅 앙상블(XGBoost, LightGBM, CatBoost)을 통해 정상/악성 및 행위 분류.

---

## 구성 요소

* **trace\_features.py**

  * JSON 트레이스를 수치 피처 벡터로 변환
  * 이벤트 카운트, 경로 토큰, 도메인 토큰 등 다양한 피처 추출

* **이진 분류 앙상블** (`train_ensemble.py`, `predict_ensemble.py`)

  * 입력: `labels.csv` (traceID, label\[0=정상,1=악성])
  * 3개 모델(LightGBM, XGBoost, CatBoost) 교차 검증 학습
  * 예측 시 각 모델의 확률 출력 및 평균 점수 기반 최종 판정

* **다중 분류 앙상블 (행위 분류)** (`train_multiclass_ensemble.py`, `predict_multiclass_ensemble.py`)

  * 입력: `behaviors.csv` (traceID, behavior\_label)
  * 동일한 3개 모델(LightGBM, XGBoost, CatBoost)을 다중 분류 모드로 학습
  * 각 모델의 클래스별 확률을 평균하여 최종 행위 유형을 결정

---

## 라벨링 관련 파일

### 1. make\_labels\_from\_folder.py

* 특정 폴더 내의 모든 `*.json` 트레이스 파일에 동일한 라벨을 부여하여 CSV로 저장
* 라벨은 **정수(이진 분류용)** 또는 **문자열(행위 분류용)** 모두 가능

**사용 예시**

```bash
# (1) 이진 분류용 라벨 생성
python make_labels_from_folder.py --data_dir traces_train/benign --label 0 --out_csv labels_benign.csv
python make_labels_from_folder.py --data_dir traces_train/malware --label 1 --out_csv labels_mal.csv

# (2) 다중 분류용 라벨 생성 (행위별 문자열 라벨)
python make_labels_from_folder.py --data_dir traces/ransomware --label ransomware --out_csv labels_ransomware.csv
python make_labels_from_folder.py --data_dir traces/trojan --label trojan --out_csv labels_trojan.csv
python make_labels_from_folder.py --data_dir traces/worm --label worm --out_csv labels_worm.csv
```

### 2. label\_2to1.py

* 여러 라벨 CSV를 합쳐서 최종 `labels.csv`(이진 분류용)와 `behaviors.csv`(다중 분류용) 생성

**동작 방식**

1. `labels_benign.csv`, `labels_mal.csv` → 병합 후 `labels.csv`
2. `labels_ransomware.csv`, `labels_trojan.csv`, `labels_worm.csv` 등 → 병합 후 `behaviors.csv`

**사용 예시**

```bash
python label_2to1.py
```

실행 후 생성 파일:

* `labels.csv` : 이진 분류 학습/예측에 사용
* `behaviors.csv` : 다중 분류 학습/예측에 사용

---

## 사전 준비

1. **Python 3.10+**
2. **필수 패키지 설치**

```bash
pip install -r requirements.txt
```

3. **데이터 준비**

   * `traces_train/`: 학습용 JSON 트레이스 (정상/악성 종합)
   * `labels.csv`: 이진 분류 라벨 파일
   * `behaviors.csv`: 행위 라벨 파일

---

## 실행 방법

### 1) 이진 분류 앙상블 학습

```bash
python train_ensemble.py --data_dir traces_train --labels_csv labels.csv --out_dir model_ens_out --n_splits 5
```

### 2) 이진 분류 예측

```bash
python predict_ensemble.py --data_dir traces_eval --model_dir model_ens_out --out_csv pred_ensemble.csv
```

### 3) 다중 분류 앙상블 학습

```bash
python train_multiclass_ensemble.py --data_dir traces_train --labels_csv behaviors.csv --out_dir model_mc_out --n_splits 5
```

### 4) 다중 분류 예측

```bash
python predict_multiclass_ensemble.py --data_dir traces_eval --model_dir model_mc_out --out_csv pred_multiclass.csv
```

