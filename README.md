## 프로젝트 소개

이 프로젝트는 화장품 제품의 성분을 분석하고 클러스터링하여 유사한 성분을 가지는 제품들을 그룹화하는 것을 목표로 합니다. 이를 통해 소비자들이 본인이 주로 사용하는 화장품의 성분과 유사한 성분이 쓰이는 화장품을 찾을 수 있고, 문제를 일으켰던 성분이 들어간 화장품을 피할 수 있을 것으로 기대합니다.

## 프로젝트 구조

```
project_directory/
│
├── data/                # 원본 데이터와 클러스터 결과들이 저장되는 디렉토리
│
├── preprocessor.py         # 데이터 수집 및 전처리 스크립트
├── recommend.py            # 클러스터링 기반 상품 추천 스크립트
├── dbscan_clustering.py    # DBSCAN 클러스터링 스크립트
├── k_means_clustering.py   # k-means 클러스터링 스크립트
├── EDA.py                  # EDA 스크립트
│
├── img/                 # 시각화 이미지 디렉토리
│
├── EDA/                 # EDA 결과 디렉토리
│
└── README.md            # 프로젝트 설명 파일
```

## 활용 데이터

네이버 쇼핑 화장품/미용 카테고리 가격 비교 탭의 총 221,700개의 상품 데이터 중에서 전처리를 거쳐 총 119,536개의 상품 데이터를 스크래핑하였습니다.

전체 데이터 중 피부에 직접 닿는 화장품인 비비크림, 클렌징, 컨디셔너, 크림, 로션, 스킨, 선크림 데이터만 추출해 클러스터링을 진행했습니다.

개별 인스턴스 구성 요소: _제품명, 제품 ID, 상품명, 제조사 및 브랜드, 상품 카테고리, 상품의 특징, 전성분, 별점, 사진_

## 데이터 전처리

1. 데이터 수집 시 상품 고유 ID가 같은 제품이거나 상품명과 제조사가 동시에 같은 제품, 상품명과 브랜드가 동시에 같은 제품인 경우 동일한 제품으로 간주하고 제거하였습니다.

2. 클러스터링을 위해 각 성분들을 one-hot 인코딩하여 범주형 변수를 수치화하였습니다.

3. `fuzzywuzzy` 패키지를 이용해 성분 문자열들 간의 유사도를 측정하고 유사도가 50 이상인 문자열을 특정 문자열로 대체하였습니다.
   
   _(fuzzywuzzy Github page: https://github.com/seatgeek/thefuzz)_

   → 예: 정제수, 정제수001 > 정제수 / 티타늄디옥사이드, 1-11티타늄디옥사이드 > 티타늄디옥사이드

## 모델 학습

### 차원 축소

고차원 데이터는 "차원의 저주"로 인해 클러스터링 알고리즘이 제대로 작동하지 않을 수 있습니다. 차원의 저주는 데이터의 차원이 증가할수록 데이터 간의 거리가 점점 비슷해지고, 이는 클러스터링 알고리즘이 효과적으로 동작하지 않게 만드는 현상입니다. 이를 해결하기 위해 차원 축소 기법을 사용합니다. 

차원 축소 기법을 사용하면 데이터의 차원을 줄여 클러스터링의 효율성을 높일 수 있습니다. 대표적인 차원 축소 기법에는 PCA, t-SNE, UMAP 등이 있습니다. 이 프로젝트에서는 주로 PCA와 t-SNE를 사용하여 차원을 축소하였습니다.

### K-means 클러스터링

최고의 성능을 내는 k값을 찾기 위해 실루엣 스코어가 가장 높은 최적의 k값을 도출하여 모델을 학습시켰습니다.

차원 축소를 위해 데이터에 PCA를 적용하여 2차원으로 축소하였습니다.

### DBSCAN

차원 축소 전 사용되는 Feature가 수백 개가 넘기 때문에 `min_sample`은 7로 고정하였습니다.

_(DBSCAN min_samples 결정 근거: https://scikit-learn.org/stable/modules/generated/dbscan-function.html)_

최고의 성능을 내는 eps 값을 찾기 위해 실루엣 스코어가 가장 높은 최적의 eps 값을 도출하여 모델을 학습시켰습니다.

차원 축소를 위해 데이터에 t-SNE를 적용하여 2차원으로 축소하였습니다. DBSCAN은 비선형적 관계를 이용해 클러스터링을 진행하기 때문에 선형적 관계를 이용해 차원 축소를 하는 PCA보단 비선형적 관계를 활용해 차원 축소를 하는 t-SNE를 사용하였습니다.

## 결과

### 분석 결과

1. DBSCAN 클러스터링은 클러스터를 분리해내지 못한 반면, k-means 클러스터링은 효과적으로 분리해 더 좋은 성능을 보였습니다.
   
2. k-means 클러스터링의 경우 k값에 따라 성능이 주기적으로 변하는 경향을 보였습니다.

3. PCA를 통해 2차원으로 차원 축소한 결과 대부분의 상품들이 k-means 클러스터링을 통해 적절한 클러스터로 분리된 것으로 보였습니다.

4. t-SNE를 통해 2차원으로 차원 축소한 결과 대부분의 상품들이 DBSCAN 클러스터링을 통해 적절한 클러스터로 분리되지 못한 것으로 보였습니다.

5. DBSCAN 클러스터링의 경우 특정 eps 이상이면 모든 상품을 이상치로 판단했습니다.

### 해석

1. k-means 클러스터링을 통해 화장품 성분을 몰라도 자신이 사용하는 화장품과 유사한 성분을 지닌 제품 그룹을 파악할 수 있습니다.

2. k-means 클러스터링을 통해 특정 상품의 다른 클러스터를 선택함으로써 자신에게 맞지 않는 성분을 지닌 제품 그룹 또한 파악할 수 있습니다.
   
3. DBSCAN은 고차원 데이터를 차원 축소한 데이터에서는 사용하기 어려운 알고리즘으로, 대부분의 경우 클러스터를 분리해내지 못했습니다.

## 한계점 및 개선방안

1. 두 클러스터링 알고리즘을 비교하기 위해 성능 평가 시 실루엣 스코어라는 하나의 성능 평가 지표만 사용했는데, 다양한 성능 평가 지표를 사용하지 못해 성능 평가가 왜곡되었을 가능성이 있습니다.

   -> 해당 데이터에서 DBSCAN 클러스터링이 효과적이지 않다는 점을 분석 결과로 알 수 있었기 때문에, 추후 분석에서는 k-means 클러스터링을 기반으로 Dunn Index와 같은 다른 클러스터링 성능 평가 지표도 사용할 예정입니다.
    
3. 현재 `recommend.py`에서는 코사인 유사도를 바탕으로 해당 상품과 가장 비슷한 상품, 가장 비슷하지 않은 상품들을 도출하고 있는데, 클러스터링 만으로는 효과적인 추천을 하는 데에 어려움이 있습니다.

   -> 추후 분석에서는 Collaborative Filtering과 같은 추천 시스템 알고리즘을 추가적으로 활용해 더 나은 추천을 제공할 예정입니다.

## 사용 방법

### 데이터 전처리

데이터 전처리를 위해 `preprocessor.py` 스크립트를 실행합니다. 이 스크립트는 원본 데이터를 읽고 필요한 전처리 작업을 수행합니다.

```bash
python preprocessor.py
```

### EDA

EDA를 위해 `EDA.py` 스크립트를 실행합니다. 이 스크립트는 EDA 결과를 다양한 그래프로 나타냅니다.

```bash
python EDA.py
```

### 클러스터링

전처리가 완료된 데이터를 사용하여 클러스터링을 수행합니다. 어떤 알고리즘을 실행하는지에 따라 `k_means_clustering.py` 또는 `dbscan_clustering.py` 스크립트를 실행하면 됩니다.

```bash
python k_means_clustering.py
# 또는
python dbscan_clustering.py
```


### 추천

클러스터링이 완료된 데이터를 바탕으로 상품 추천을 위해 `recommend.py` 스크립트를 실행합니다.

```bash
python recommend.py
```
