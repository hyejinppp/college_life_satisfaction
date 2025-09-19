
---

# 📚 대학생활 만족도 예측 모델링 프로젝트(sat2.ipynb)

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

*결측치 컬럼 제거: 분석에 의미가 없거나 결측치가 많은 변수 삭제 (drop_missing_cols)
*오타 수정: birth_area의 잘못된 값('g')을 'gangwon'으로 변경
*범주형 변수 인코딩:
dance_years → 숫자 매핑 (less2=1, 2to4=2, …)
기타 범주형 변수 → 원-핫 인코딩 (첫 번째 카테고리 제거, drop_first=True)
분석 제외 변수 제거: 모델링에서 제외할 변수 삭제 (prof_hi, int_major, enter_year)
상관계수 높은 변수 사전 삭제: 다중공선성 예방 (lecture_qual, peer_personal, current_area...)
*타깃 변수 생성:sat = 4개 만족도 변수 평균
sat_group = 중위수 기준 이진 분류 (1: 상위 만족, 0: 하위 만족)

* **출력**:

  * 최종 컬럼 수, `sat` 및 `sat_group` 확인

✅ 모든 전처리는 재실행 시 안전하며, 시드(RANDOM_STATE=42)가 통일
---

### 2.2 변수 선택 + RFECV / ElasticNet 파이프라인 (2번 코드)

* **설명**:

  * ElasticNet (LogisticRegressionCV) / RFECV(LR, DT, RF, XGB) 파이프라인 정의
  * 훈련 데이터 내부에서만 변수 선택 → 안정적인 변수 탐색
  * 폴드별 선택 변수 → 교집합(안정적 변수) 추출
  * 모델 성능: CV(교차검증) & 홀드아웃 평가
  * **중요**: 변수 선택은 X\_train 내부 fold에서만 수행, 정보 누수 없음
  * Outer CV (5-fold) + Fold별 선택 변수 → 안정적 변수 교집합 산출
  * Jaccard 지수 계산 → 변수 선택 안정성 평가

* **세부설명**:
  
  * 데이터 분할: train_test_split + Stratified CV
  * 파이프라인 정의:
  * ElasticNet: StandardScaler → LogisticRegressionCV (l1_ratios=[.1,.3,.5,.7,.9])
  * RFECV: Base estimator → RFECV (feature selection) → 최종 estimator
  * RFECV: min_features_to_select = 5% of X_train, scoring='f1'
  * n_jobs=-1 (병렬), 중첩 CV 고려

   * 폴드별 변수 선택: 각 outer CV fold에서 학습 후 선택 변수 추출
   * 교집합 → 안정적 변수
   * Jaccard 안정성 지수 계산: 각 fold 선택 변수 간 유사도 평균 → 변수 선택 안정성 평가
   
   * 모델 성능 평가:CV 평균 및 표준편차,Hold-out 테스트셋 성능 (accuracy, f1, precision, recall, roc_auc)
   *
   * 최종 선택 변수 확인:RFECV: support_ 사용, ElasticNet: 비영(0) 계수 변수 사용
   * 
* **출력**:

  * 폴드별 선택 변수 수
  * 안정적 변수 (교집합)
  * Jaccard안정성지수
  * 최종 파이프라인에서 선택된 변수

---

### 2.3 안정적 변수 기반 최종 모델링 (3번 코드)

* **설명**:

  * RFECV / ElasticNet 선택 후 교집합 안정적 변수 사용
  * LogisticRegression, DecisionTree, RandomForest, XGBoost->LogisticRegression만 StandardScaler 적용
  * 5-Fold CV: 안정적 변수 기반 교차검증
  * Hold-out 평가: X_test에 대해 accuracy, balanced accuracy, F1, precision, recall, ROC-AUC, MCC 계산
    
* **출력**:

  * 5-fold CV + Hold-out 평가
  * 성능 지표: Accuracy, F1, Precision, Recall, ROC-AUC
  * 최종 성능 표

---

### 2.4 CV F1 기준 상위 5개 모델 상세 성능 (4번 코드)

* **설명**:

  * CV f1기준 상위 5개 모델을 기반으로 상세 CV 및 테스트 평가
  * 혼동행렬(Confusion Matrix) 출력
* **출력**:

  * CV / Test F1, Accuracy, Precision, Recall, ROC-AUC
  * CV-Test F1 gap 확인
  * 혼동행렬

  * 상세 성능 결과: 전체적인 분류성능과 과적합 반응을 보이지 않는 RFECV\_RF + XGB 조합 최적 모델로 확인
---

### 2.5 최적 모델 SHAP 변수 중요도 계산 (5번 코드)

* **설명**:

  * RFECV\_RF + XGB Pipeline 기반
  * TreeExplainer로 SHAP 값 계산
  * 변수별 절대값 평균, 평균 SHAP 값 산출
* **출력**:

  * 변수별 SHAP 수치 데이터프레임 (`feature`, `shap_mean_abs`, `shap_mean`)
  * 상위 N개 변수 확인 가능

---

### 2.6 최적 모델 SHAP 및 ROC 시각화 (6번 코드)

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
