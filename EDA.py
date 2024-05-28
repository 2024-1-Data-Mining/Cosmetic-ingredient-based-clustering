import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from matplotlib import font_manager, rc

def EDA(category):
    # 데이터 로드
    file_path = f'data/{category}.xlsx'
    df = pd.read_excel(file_path)

    # 성분 데이터를 리스트로 변환
    df['성분_리스트'] = df['성분'].apply(ast.literal_eval)

    # 각 성분의 빈도 분석
    all_ingredients = [ingredient for sublist in df['성분_리스트'] for ingredient in sublist]
    ingredient_counts = Counter(all_ingredients)

    # 상위 20개의 성분 빈도수 시각화
    top_20_ingredients = ingredient_counts.most_common(20)
    top_20_df = pd.DataFrame(top_20_ingredients, columns=['Ingredient', 'Count'])

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Count', y='Ingredient', data=top_20_df)
    plt.title('Top 20 Most Common Ingredients')
    plt.savefig(f'EDA/EDA_{category}_Top 20 Most Common Ingredients.png')

    # 각 제품의 성분 수 분포 시각화
    df['성분_수'] = df['성분_리스트'].apply(len)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['성분_수'], bins=30, kde=True)
    plt.title('Distribution of Number of Ingredients per Product')
    plt.xlabel('Number of Ingredients')
    plt.ylabel('Frequency')
    plt.savefig(f'EDA/EDA_{category}_Distribution of Number of Ingredients per Product.png')

if __name__ == '__main__':
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 한글 깨짐 현상 방지를 위한 폰트 설정
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    category = 'suncream' # category 여기서 변경, 전체 데이터 분석할 경우 'data' 입력
    EDA(category)