
# University Life Satisfaction Prediction for Dance Major Students

무용전공 대학생 355명을 대상으로 대학생활 만족도(`sat`)를 예측하고, 저만족 위험군을 분류하기 위한 머신러닝 기반 분석 프로젝트입니다.

본 프로젝트는 데이터 누수(data leakage)를 방지한 변수선택, 안정 변수셋(stable feature set) 도출,30회 반복 홀드아웃 기반 모델 비교, 최종 테스트셋 평가 및 시각화까지 포함한 전체 파이프라인으로 구성되어 있습니다.

## 1. 연구 목적

이 프로젝트의 목적은 다음과 같습니다.

- 대학생활 만족도 연속지표 `sat`를 구성
- `sat`를 기반으로 저만족군(1) / 고만족군(0) 분류
- 변수선택 과정에서도 테스트 데이터가 개입되지 않도록 설계
- 반복적인 split을 통해 안정적으로 선택되는 변수 도출
- 여러 모델 조합을 비교하여 최적 분류모델 선정
- 최종적으로 테스트 성능, 혼동행렬, ROC, OR, Beta, VIF 등을 종합 보고

## 2. 데이터

- 데이터 파일: `univ355.csv`
- 대상: 무용전공 대학생 355명
- 만족도 구성 문항:
  - `univ_proud`
  - `univ_belong`
  - `major_proud`
  - `major_belong`

위 4개 문항의 평균으로 연속형 만족도 지표 `sat`를 생성합니다.

## 3. 전체 분석 흐름

### Step 1. 전처리 및 sat 생성
- 대량 결측 또는 의미가 약한 변수 제거
- 지역 관련 중복 변수 제거
- 오타 범주값 정정
- 순서형 변수(`dance_years`) ordinal coding
- 범주형 변수 원-핫 인코딩
- 다중공선성 우려 변수 일부 제거
- 최종적으로 `sat` 생성

### Step 1-2. 차이검정 및 사후검증
- 이진형 변수: t-test
- 다범주형 변수: ANOVA
- 유의한 다범주형 변수에 대해 Tukey HSD 사후검증 수행

### Step 2. 변수선택을 포함한 6:2:2 파이프라인
반복마다 다음 절차를 수행합니다.

1. `train_val/test = 80/20`
2. `train/val = 75/25` → 최종 6:2:2
3. `train` 데이터의 median으로 threshold(`thr`) 생성
4. `sat < thr` 이면 저만족(1), 아니면 고만족(0)
5. 변수선택은 오직 Train(60%)에서만 수행
6. 선택된 변수셋의 품질은 Val(20%)에서만 평가
7. 각 반복에서 가장 좋은 변수선택 방법 기록

사용한 변수선택 방법:
- ElasticNet
- RFECV(LR)
- RFECV(DT)
- RFECV(RF)
- RFECV(XGB)

### Step 2-2. 안정 변수셋(stable feature set) 생성
30회 반복 중 25회 이상 선택된 변수만 안정 변수로 정의합니다.

대표 안정 변수셋:
- `vars_enet_k25`
- `vars_rfecv_rf_k25`
- `vars_rfecv_xgb_k25`

### Step 3-1. 12개 조합 모델 비교(참고용-안해도 됨)
안정 변수셋 3개 × 모델 4개 = 총 12개 조합 비교

변수셋:
- ElasticNet(K25)
- RFECV(RF)(K25)
- RFECV(XGB)(K25)

모델:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

비교 기준:
1. `Val_recall_1`
2. `Val_f1_1`
3. `Val_roc_auc`

여기서 class 1은 저만족군입니다.

### Step 3-2. ElasticNet(K25) 전용 4개 모델 비교
ElasticNet(K25) 변수셋만 따로 사용하여 아래 4개 모델을 다시 비교합니다.

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Step 4. 최적모델 최종 테스트 평가
최종 선택된 모델은 ElasticNet(K25) + Logistic Regression 조합입니다.

이 모델에 대해 8:2 반복 홀드아웃 30회를 수행하여 테스트셋 기준으로 최종 성능을 평가합니다.

평가 및 시각화 항목:
- Recall / Precision / F1 / ROC-AUC
- 누적 혼동행렬
- OR forest plot + 95% CI
- 특성 간 상관행렬
- 평균 ROC curve
- Beta coefficient plot
- VIF 포함 최종 보고표

## 4. 데이터 누수 방지 원칙

이 프로젝트의 핵심은 데이터 누수 방지입니다.

- `sat_group`를 전처리 단계에서 미리 만들지 않음
- threshold는 각 반복의 train 데이터 median으로만 계산
- 스케일링은 train 데이터로만 학습
- 변수선택도 train 데이터에서만 수행
- test 데이터는 최종 평가 단계 전까지 사용하지 않음

## 5. 주요 평가 지표

본 분석은 저만족군(1) 탐지를 중요하게 다룹니다.

- `Recall_1`: 실제 저만족군을 얼마나 잘 찾아냈는가
- `Precision_1`: 저만족으로 예측한 대상 중 실제 저만족 비율
- `F1_1`: 저만족군 기준 정밀도와 재현율의 조화평균
- `ROC_AUC`: 전체 판별력
- `Balanced Accuracy`: 클래스 불균형을 고려한 정확도
- `F1_macro`: 클래스 전체 평균 F1

반복 내 최적 조합 선정 우선순위:
1. `Val_recall_1`
2. `Val_f1_1`
3. `Val_roc_auc`

## 6. 실행 순서

1. 전처리 + `sat` 생성 + 차이검정
2. 차이검정 사후검증
3. 변수선택 파이프라인 실행
4. 안정 변수셋 생성
5. 12조합 모델 비교(안해도 됨)
6. ElasticNet(K25) 전용 4개 모델 비교
7. 최적모델 최종 테스트 평가 및 시각화

## 7. 필요 패키지

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- xgboost

## 8. 주요 산출물

표 형태 결과:
- t-test / ANOVA 결과표
- Tukey HSD 사후검증 결과
- 변수선택 방법별 Val 성능 요약표
- 안정 변수셋 목록
- 12조합 모델 비교표
- ElasticNet(K25) 전용 4개 모델 비교표
- OR + 95% CI + VIF 보고표
- 최종 테스트 성능 요약표

그림 형태 결과:
- 누적 혼동행렬
- Odds Ratio forest plot
- 상관행렬 heatmap
- 평균 ROC curve
- Beta coefficient plot

## 9. 최종 모델

- Feature set: `ElasticNet(K25)`
- Classifier: `LogisticRegression`
- Final evaluation: 8:2 repeated hold-out (30회)
- Positive class: 저만족 = 1

즉, 최종 목표는 저만족 위험군을 얼마나 안정적으로 탐지할 수 있는가에 초점을 둡니다.

## 10. 한 줄 요약

이 프로젝트는 무용전공 대학생의 대학생활 만족도 저하 위험군을 데이터 누수 없이 안정적으로 예측하기 위한 반복 홀드아웃 기반 머신러닝 파이프라인입니다.
