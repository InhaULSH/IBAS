# 9. 지도 학습 - 분류
# %% 주어진 피쳐와 레이블 값을 머신러닝 알고리즘으로 학습 => 모델 생성 => 이후 주어지는 피쳐에 대해 레이블 값 예측
# 결정 트리 : 분류 모델 중 하나, 유연성 높음 / 사전 가공의 영향 적음 / 과적합 위험성 존재 / 앙상블 기법에 적합
# 데이터를 특정 규칙에 따라 분류, 해당 규칙을 바탕으로 트리 기반 분류 모델 생성, 깊이 조절하여 과적합 예방 필요
# 결정트리의 성능은 정보이득과 엔트로피를 기준으로 평가, 정보이득 최대화 또는 엔트로피 최소화하는 규칙이 가장 이상적
# 앙상블 기법 : 여러 개의 저성능 학습기를 결합해 오류에 대한 가중치를 업데이트하며 예측성능을 향상 시키는 기법
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = {
    'Weather': ['Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Warm', 'Cool', 'Warm', 'Cool', 'Warm'],
    'PlayTennis': ['Yes', 'No', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
# 데이터 준비
df_encoded = pd.get_dummies(df[['Weather', 'Temperature']])
df_encoded['PlayTennis'] = df['PlayTennis'].apply(lambda x: 1 if x == 'Yes' else 0)
# 데이터 인코딩
X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']
model = DecisionTreeClassifier(max_depth=1)
model.fit(X, y)
# 학습 데이터 분할 및 학습
from sklearn.tree import export_graphviz
import graphviz
# feature_names: 문제(X)의 컬럼 이름들
# class_names: 정답(y)의 이름 ('안 친다', '친다')
dot_data = export_graphviz(model,
                           feature_names=X.columns,
                           class_names=['Yes', 'No'],
                           filled=True)
graph = graphviz.Source(dot_data)
graph.render('decision_tree', format='png', view=True)
print("decision_tree.png 파일로 저장이 완료되었습니다.")
# 결정트리 시각화

# %% 과적합 : 학습 데이터의 미세한 변동까지 학습하여 모델이 지나치게 복잡해지고 평가 데이터 일반화 성능은 하락
# 학습 정확도는 개선되지만 평가 정확도는 악화되는 지점이 과적합 발생 지점, 하이퍼파라미터 튜닝을 통해 과적합 억제 가능
# 결정트리의 하이퍼 파라미터 = max_depth / min_samples_split / min_samples_leaf / criterion / splitter
# max_depth : 트리의 최대 깊이, 작게 할수록 모델이 단순해지며 과적합을 억제하는 정도가 강해짐
# min_samples_split : 분할을 위한 최소 샘플수, 지나치게 소수의 데이터로 규칙을 생성하는 것을 억제함
# min_samples_leaf : 리프 노드의 최소 샘플수, 생성되는 리프 노드의 표본수가 일정 개수 이상이어야 분할을 실시함
# criterion : 분할의 성능평가 기준, gini 또는 entropy, 두 가지 모두 테스트해보고 성능이 좋은 것 선택
# splitter : 각 노드에서 최적 분할 규칙 결정 기준, best(전수조사) 또는 random(무작위 표본 조사),
# 품질은 best가 더 우수하나 속도가 느리고 random이 속도가 더 빠르며 다소의 과적합 억제 효과가 있음

# %%피쳐 중요도 = 모델이 예측을 할 때 각각의 피처들이 얼마나 중요하게 작용했는지를 숫자로 나타낸 값, 피쳐의 기여도
# 해당 피쳐를 기준으로 분류했을때 엔트로피가 최소화되는 정도, 정보이득이 얼마나 큰 지를 기준으로 계산
import seaborn as sns
import matplotlib.pyplot as plt
importances = model.feature_importances_
# model.feature_importances_ 로 중요도 확인
feature_importances = pd.Series(importances, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
# 보기 쉽게 데이터프레임으로 만들기
plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()
print(feature_importances)
# 시각화

# %% UCI 사용자 행동 인식 데이터셋
# 피처 이름 파일 읽기
feature_name_df = pd.read_csv("./UCI HAR Dataset/features.txt", header=None, names=['column_index', 'column_name'])
feature_names = feature_name_df.iloc[:, 1].values.tolist()
# 훈련 데이터 불러오기
X_train = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", names=feature_names)
y_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", header=None, names=['action'])
# 테스트 데이터 불러오기
X_test = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", names=feature_names)
y_test = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", header=None, names=['action'])
# 데이터 확인
print('훈련 데이터셋 모양:', X_train.shape)
print('테스트 데이터셋 모양:', X_test.shape)
X_train.head()