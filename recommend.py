import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar_products(product_ID, df, num_recommendations=5):
    # 화장품 이름을 제외한 성분 데이터만 추출
    ingredient_data = df.drop(columns=['ID', '상품명', '브랜드', '이미지'])

    # 특정 제품의 클러스터 찾기
    product_cluster = df[df['ID'] == product_ID]['Cluster'].values[0]
    
    # 코사인 유사도를 계산하여 가장 유사한 제품 추천
    product_idx = df[df['ID'] == product_ID].index[0]
    cosine_sim = cosine_similarity(ingredient_data)
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 같은 클러스터에 있는 제품들 중에서 유사한 제품 선택 (자기 자신 제외)
    similar_product_indices = [i[0] for i in sim_scores if df.iloc[i[0]]['Cluster'] == product_cluster and i[0] != product_idx]
    similar_products = df.iloc[similar_product_indices][['ID', '상품명', '이미지']].head(num_recommendations)
    
    return similar_products

def recommend_dissimilar_products(product_ID, df, num_recommendations=5):
    # 화장품 이름을 제외한 성분 데이터만 추출
    ingredient_data = df.drop(columns=['ID', '상품명', '브랜드', '이미지'])

    # 특정 제품의 클러스터 찾기
    product_cluster = df[df['ID'] == product_ID]['Cluster'].values[0]
    
    # 코사인 유사도를 계산하여 가장 유사하지 않은 제품 추천
    product_idx = df[df['ID'] == product_ID].index[0]
    cosine_sim = cosine_similarity(ingredient_data)
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1])
    
    # 다른 클러스터에 있는 제품들 중에서 유사하지 않은 제품 선택
    dissimilar_product_indices = [i[0] for i in sim_scores if df.iloc[i[0]]['Cluster'] != product_cluster]
    dissimilar_products = df.iloc[dissimilar_product_indices][['ID', '상품명', '이미지']].head(num_recommendations)
    
    return dissimilar_products

if __name__ == '__main__':
    file_path = 'data/k_means_lotion.xlsx' # 사용할 데이터 설정
    df = pd.read_excel(file_path)

    product_id = 24364546525  # 추천할 기준이 되는 화장품 ID 입력, 예시로 1025 독도 로션 400ml 사용
    similar_products = recommend_similar_products(product_id, df)
    dissimilar_products = recommend_dissimilar_products(product_id, df)

    print(f"'{product_id}'와 가장 비슷한 상품 5개:")
    for index, row in similar_products.iterrows():
        print(f"ID: {row['ID']}, 상품명: {row['상품명']}, 이미지: {row['이미지']}")

    print(f"'{product_id}'와 가장 비슷하지 않은 상품 5개:")
    for index, row in dissimilar_products.iterrows():
        print(f"ID: {row['ID']}, 상품명: {row['상품명']}, 이미지: {row['이미지']}")