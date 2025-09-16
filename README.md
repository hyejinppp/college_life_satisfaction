# College Life Satisfaction Analysis

이 프로젝트는 대학생활 만족도를 분석하고 예측 모델을 구축하기 위한 과정입니다.

## 프로젝트 개요
# 타깃
- 대학생활 만족도(sat)를 4가지 변수(univ_proud, univ_belong, major_proud, major_belong)평균값으로 계산
- sat의 중위수 기준으로 **고만족 / 저만족** 그룹으로 분류

# 예측변수
- 선택 변수(feature selection) 과정을 거쳐 모델 입력 변수 선정
- ->rfecv(lr),elasticnet 선택변수가 jaccard 가장 높으며 변수 수도 적절함
- vif, 상관계수로 다중공선성 검증

# 모델링
- 다양한 분류 모델을 학습하고 성능 비교
- rfecv(lr)변수+logistic regression모델과 elasticnet변수+randomforest모델이 최적 모델 후보
- 최적의 모델을 선정하여 대학생활 만족도 예측
- -rfecv(lr)logistic regression모델 최적모델


## 파일 구성
- `Sat/sat.ipynb` : 분석 및 모델링 노트북
- `Sat/univ355.csv` : 데이터 파일

## 사용 방법
1. GitHub에서 파일을 다운로드
2. Jupyter Notebook 또는 Google Colab에서 `sat.ipynb` 열기
3. 필요 시 데이터 파일 경로 확인 후 실행
