import pandas as pd
from ast import literal_eval
from fuzzywuzzy import process, fuzz
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA

def one_hot_encoding(category):
    df = pd.read_excel(f'data/{category}.xlsx')
    df['성분'] = df['성분'].apply(literal_eval)

    # 모든 성분을 추출하여 리스트로 변환
    all_ingredients = []
    df['성분'].apply(lambda x: all_ingredients.extend(x))

    # 고유 성분 사전 생성
    unique_ingredients = get_unique_ingredients(all_ingredients)

    # 데이터프레임의 성분 열을 고유 성분명으로 변환
    df['성분'] = df['성분'].apply(lambda x: replace_with_unique(x, unique_ingredients))

    # 모든 성분을 집합으로 추출
    all_ingredients = set()
    for ingredients in df['성분']:
        all_ingredients.update(ingredients)

    # 성분 리스트를 정렬된 상태로 변환
    all_ingredients = sorted(all_ingredients)

    # 결과를 저장할 데이터프레임을 생성
    ingredient_data = []

    for ingredients in df['성분']:
        row = {}
        for ingredient in all_ingredients:
            row[ingredient] = 1 if ingredient in ingredients else 0
        ingredient_data.append(row)

    # 데이터프레임으로 변환
    df_ingredients = pd.DataFrame(ingredient_data)

    df = df.loc[:, ['ID','상품명','브랜드','이미지']]
    df = pd.concat([df, df_ingredients], axis=1)

    return df

# 중복 성분 제거를 위한 기준 성분 설정
def get_unique_ingredients(ingredients):
    unique_ingredients = {}
    for ing in ingredients:
        # 기존 성분들과 유사도 비교
        match = process.extractOne(ing, unique_ingredients.keys(), scorer=fuzz.token_set_ratio)
        if match and match[1] > 50:  # 유사도가 50 이상
            unique_ingredients[match[0]].append(ing)
        else:
            unique_ingredients[ing] = [ing]
    return unique_ingredients

# 고유 성분명을 기준으로 성분 리스트 변환
def replace_with_unique(ingredient_list, unique_ingredients):
    new_list = []
    for ing in ingredient_list:
        if not ing:  # 빈 문자열 체크
            continue
        match = process.extractOne(ing, unique_ingredients.keys(), scorer=fuzz.token_set_ratio)
        if match and match[1] > 50:  # 유사도가 50 이상
            new_list.append(match[0])
        else:
            new_list.append(ing)
    return new_list

def k_means_clustering(category, df):
    # 화장품 이름을 제외한 성분 데이터만 추출
    ingredient_data = df.drop(columns=['ID', '상품명', '브랜드', '이미지'])

    k_range = range(2, 11)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(ingredient_data)
        score = silhouette_score(ingredient_data, labels)
        silhouette_scores.append(score)
    
    # 최적의 K 값 찾기 (실루엣 점수가 가장 높은 K 값 사용)
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    
    # 실루엣 점수 시각화
    plt.figure(1)
    plt.plot(k_range, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f"{category} cluster's Silhouette score")
    plt.title('Silhouette score for different k values')
    plt.savefig(f'img/k_means_{category}_silhouette_score.png')

    print(f'이 클러스터링에서 최적의 k는 {optimal_k}이고, silhouette score는 {max(silhouette_scores)}이다.')

    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(ingredient_data)

    # 각 제품의 Cluster label을 데이터프레임에 추가
    df['Cluster'] = kmeans.labels_

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(ingredient_data)

    # PCA 결과 데이터프레임에 추가
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]

    # 클러스터 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
    plt.title(f'Cosmetic Products({category}) Clustering')
    plt.savefig(f'img/k_means_{category}_visualization.png')

    df.to_excel(f'data/k_means_{category}.xlsx')
    print(f'{category} K-means clustering 분석 완료')

if __name__ == '__main__':
    category = 'cream'  # category 여기서 변경
    df = one_hot_encoding(category)
    k_means_clustering(category, df)