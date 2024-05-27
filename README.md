# Cosmetic-ingredient-based-clustering

## 데이터 획득 및 설명
네이버 쇼핑몰의 화장품/미용 카테고리 일부 품목 크롤링

개별 instance 구성요소
_제품명, 제품 아이디, 상품명, 제조사 및 브랜드, 상품 카테고리, 상품의 특징, 전성분, 별점, 사진_

## 데이터 전처리
성분 변수 제외한 나머지 변수 제거
→ 개별 성분들을 독립 변수로 사용

클러스터링 위해 각 성분들 one→hot 인코딩
→ 범주형 변수 수치화

중복된 변수 제거
→ ex. 정제수, 정제수001 -> 정제수

성분들간의 유사도를 기준으로 고유한 성분 결정
+ 유사한 성분들은 그룹핑하여 하나의 성분으로 처리

## 모델 학습

### K-means 클러스터링
실루엣 스코어가 가장 높은 최적의 k값 도출하여 모델 학습
시각화 위해 데이터에 PCA 적용하여 2차원으로 축소

### DBSCAN
실루엣 스코어가 가장 높은 최적의 eps값 도출하여 모델 학습
시각화 위해 데이터에 PCA 적용하여 2차원으로 축소

DBSCAN min_samples 결정 근거: https://scikit-learn.org/stable/modules/generated/dbscan-function.html

## 분석 결과
코드 마저 올라오면추가예정

### 해석 및 시사점
- 화장품 성분 몰라도 자신이 사요하는 화장품과 유사한 성분 지닌 제품 그룹 파악 가능
- 자신에게 맞지 않는 성분 지닌 제품 그룹 또한 파악 가능


## 분석 한계점

- K-means 클러스터링은 이상치에 민감하고, DBSCAN은 밀도 차이가 클 경우 잘못된 분석 결과가 나올 수 있음
→ 사용한 모델들을 앙상블 해서 학습시키거나, 이상치에 덜 민감한 알고리즘을 사용했다면 더 좋은 결과가 나왔을 것 같음
- 성분이 적혀있지 않은 data들을 제거하는 과정에서 생각보다 많은 데이터가 날아감
→ 더 많은 데이터들을 크롤링해 모델 학습에 사용했다면 보다 나은 성능을 보였을 것 같음
