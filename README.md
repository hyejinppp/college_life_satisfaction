
---

# 📚 대학생활 만족도 예측 모델링 프로젝트

본 프로젝트는 \*\*대학생활 만족도(SAT)\*\*를 이진 분류로 예측하기 위한 데이터 전처리, 변수 선택, 모델링, 성능 평가 및 SHAP 분석을 포함합니다.

---

## 1️⃣ 프로젝트 개요

* **목적**: 실기예체능 전공 대학생의 대학생활 만족도 예측
* **데이터**: `univ355.csv`
* **타깃 변수**: `sat_group` (중위수 기준 만족도 이진화)
* **분석 흐름**:

  1. 데이터 전처리 및 결측치/불필요 변수 제거
  2. 범주형 변수 인코딩 (`get_dummies(drop_first=True)`)
  3. 변수 선택 (ElasticNet, RFECV 기반)
  4. 교차검증 및 홀드아웃 평가
  5. 안정적 변수 기반 최종 모델 학습
  6. SHAP 분석 및 ROC Curve 시각화

---

## 2️⃣ 코드 구성 안내

### 2.1 전처리 및 데이터 준비 (1번 코드)

* **설명**:

  * 결측치 컬럼 제거
  * 오타 수정 (`birth_area` 등)
  * 범주형 변수 원-핫 인코딩 (`drop_first=True`)
  * 분석 제외 변수 제거
  * 상관계수 높은 변수 사전 제거
  * `sat` 변수 생성 후 median 기준 이진화 → `sat_group`
* **출력**:

  * 최종 컬럼 수, `sat` 및 `sat_group` 확인

✅ 모든 전처리는 재실행 시 안전하며, 시드(RANDOM_STATE=42)가 통일
---

### 2.2 변수 선택 + RFECV / ElasticNet 파이프라인 (2번 코드)

* **설명**:

  * ElasticNet (LogisticRegressionCV) / RFECV(LR, DT, RF, XGB) 파이프라인 정의
  * **중요**: 변수 선택은 X\_train 내부 fold에서만 수행, 정보 누수 없음
  * Outer CV (5-fold) + Fold별 선택 변수 → 안정적 변수 교집합 산출
  * Jaccard 지수 계산 → 변수 선택 안정성 평가
* **출력**:

  * 폴드별 선택 변수 수
  * 안정적 변수 (교집합)
  * 최종 파이프라인에서 선택된 변수

---

### 2.3 안정적 변수 기반 최종 모델링 (3번 코드)

* **설명**:

  * RFECV / ElasticNet 선택 후 교집합 안정적 변수 사용
  * LogisticRegression, DecisionTree, RandomForest, XGB 적용
  * 5-fold CV + Hold-out 평가
* **출력**:

  * 5-fold CV + Hold-out 평가
  * 성능 지표: Accuracy, F1, Precision, Recall, ROC-AUC
  * 최종 성능 표

---

### 2.4 상위 3개 모델 안정적 변수 상세 성능 (4번 코드)

* **설명**:

  * 상위 3개 모델에서 선택된 안정적 변수를 기반으로 상세 CV 및 테스트 평가
  * 혼동행렬(Confusion Matrix) 출력
* **출력**:

  * CV / Test F1, Accuracy, Precision, Recall, ROC-AUC
  * CV-Test F1 gap 확인
  * 혼동행렬

---

### 2.5 SHAP 변수 중요도 계산 (5번 코드)

* **설명**:

  * RFECV\_RF + XGB Pipeline 기반
  * TreeExplainer로 SHAP 값 계산
  * 변수별 절대값 평균, 평균 SHAP 값 산출
* **출력**:

  * 변수별 SHAP 수치 데이터프레임 (`feature`, `shap_mean_abs`, `shap_mean`)
  * 상위 N개 변수 확인 가능

---

### 2.6 SHAP 및 ROC 시각화 (6번 코드)

* **설명**:

  * ROC Curve 시각화 (`RocCurveDisplay`)
  * SHAP 시각화:

    * Bar plot (절대값 기준 중요도)
    * Summary dot plot
    * Summary bar plot
* **출력**:

  * ROC Curve
  * Top 변수 Bar plot
  * SHAP summary plots

---

## 3️⃣ 실행 순서

1. `1번 코드` → 데이터 전처리 완료
2. `2번 코드` → 변수 선택, 안정적 변수 추출
3. `3번 코드` → 안정적 변수 기반 모델 학습 및 평가
4. `4번 코드` → 상위 모델 상세 평가, 혼동행렬 확인
5. `5번 코드` → SHAP 계산 (변수 중요도 확인)
6. `6번 코드` → SHAP 및 ROC 시각화

> **Tip**: 전체 실행 시 `RANDOM_STATE = 42` 적용되어 재실행 가능, 정보 누수 없음

---

### 4️⃣ 요구 패키지

```bash
pip install numpy pandas scikit-learn xgboost statsmodels shap matplotlib
```

### 특이사항: RFECV_RF선택변수 75개중 아래 7가지 변수는 XGB 트리에 사용되지 않았다. 

dance_years
drink_freq
drop_out
edu_doubt
exp_group_comp
exp_solo_comp
housing_culture 
->XGB가 학습 과정에서 이 7개 변수로는 더 이상 정보 이득(Information Gain)을 얻을 수 없다고 판단
 즉, 7개 변수는 XGB가 예측을 위해 “불필요”하다고 판단한 변수
