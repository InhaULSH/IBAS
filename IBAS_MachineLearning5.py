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

# %% 피쳐 중요도 = 모델이 예측을 할 때 각각의 피처들이 얼마나 중요하게 작용했는지를 숫자로 나타낸 값, 피쳐의 기여도
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
# 작업 디렉토리 설정
import os
os.chdir('C:/Users/LSH/PycharmProjects/IBAS')
print("작업 디렉토리:", os.getcwd())
# 피처 이름 파일 읽기
def _make_unique(names):
    seen = {}
    unique = []
    for n in names:
        if n in seen:
            seen[n] += 1
            unique.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            unique.append(n)
    return unique
feature_name_df = pd.read_csv("./UCI HAR Dataset/features.txt", sep=r"\s+" , header=None, names=['column_index',
                                                                                                 'column_name'], engine="python")
feature_names = _make_unique(feature_name_df["column_name"].tolist())
# 훈련 데이터 불러오기
X_train = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", sep=r"\s+" , names=feature_names, engine="python")
y_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", sep=r"\s+" , header=None, names=['action'], engine="python")
# 테스트 데이터 불러오기
X_test = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", sep=r"\s+" , names=feature_names, engine="python")
y_test = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", sep=r"\s+" , header=None, names=['action'], engine="python")
# 데이터 확인
print('훈련 데이터셋 모양:', X_train.shape)
print('테스트 데이터셋 모양:', X_test.shape)
X_train.head()
y_train.value_counts()
# 데이터 학습
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'결정 트리 예측 정확도: {accuracy:.4f}')
# 최적 최대 깊이 튜닝
depth_accuracies = []
depth_range = range(2, 21)
for depth in depth_range:
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=156)
    dt_clf.fit(X_train, y_train)

    # 예측 및 정확도 계산
    y_pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 결과 저장
    depth_accuracies.append(accuracy)
    print(f'max_depth = {depth}, Accuracy = {accuracy:.4f}')
best_depth_index = depth_accuracies.index(max(depth_accuracies))
best_depth = depth_range[best_depth_index]
best_accuracy = max(depth_accuracies)
print(f'\nBest Accuracy: {best_accuracy:.4f} at max_depth = {best_depth}')
plt.figure(figsize=(10, 6))
plt.plot(depth_range, depth_accuracies, marker='o')
plt.title('Accuracy vs. max_depth for Decision Tree')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.xticks(depth_range)
plt.grid(True)
plt.show()
# 최적 최대 깊이로 학습 진행 및 상위 10개 피쳐 기여도 확인
dt_clf_final = DecisionTreeClassifier(max_depth=8, random_state=156)
dt_clf_final.fit(X_train, y_train)
y_pred = dt_clf_final.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'결정 트리 예측 정확도: {accuracy:.4f}')
importances = dt_clf_final.feature_importances_
feature_importances = pd.Series(importances, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False)[:10]
plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()
print(feature_importances)

# %%앙상블 기법 : 여러 개의 분류기를 생성하고 그 예측을 결합, 보팅 / 배깅 / 부스팅 / 스태킹 등
# 단일 모델의 약점 보완 가능, 결정 트리 기반, 결정 트리의 과적합 억제 및 분류 직관성 강화하는 역할 가능
# %% 보팅 : 서로 다른 여러 개의 분류기로 학습 후 예측 결과를 가지고 다수결로 결정
# 여러 알고리즘 + 하나의 데이터셋 사용 : 데이터셋 -> 모델1, 모델2, 모델3... -> 가장 많이 예측된 결과값으로 결정
# 하드 보팅 = 단순 다수결 vs 소프트 보팅 = 예측한 클래스의 확률을 평균내어 확률이 가장 높은 쪽을 채택
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
lr_clf = LogisticRegression(max_iter=5000)
knn_clf = KNeighborsClassifier(n_neighbors=8)
dt_clf = DecisionTreeClassifier(max_depth=8, random_state=156)
vo_clf = VotingClassifier(
    estimators=[('LR', lr_clf), ('KNN', knn_clf), ('DT', dt_clf)],
    voting='hard'
)
vo_clf.fit(X_train, y_train.values.ravel())
y_pred = vo_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'보팅(Voting) 분류기 정확도: {accuracy:.4f}')
# %% 배깅 : 데이터로부터 무작위 복원추출하여 여러개의 샘플을 생성 후 동일한 분류기에 학습시킨 다음 이를 종합하여 최종 결론 채택
# 하나의 알고리즘 + 여러 데이터셋 사용 : 샘플1, 샘플2, 샘플3... -> 모델 -> 예측된 결과값 평균내어 최종 결과값 결정
# 랜덤 포레스트 = 분류기로 결정트리 사용, 전체 피쳐 대신 랜덤하게 선정된 일부 피쳐만 고려, 배깅 + 결정트리 + 피쳐 샘플링
# 랜덤 포레스트의 하이퍼 파라미터 = n_estimators / max_features / max_samples
# n_estimators : 포레스트를 구성하는 결정트리의 개수, 높아질수록 대체로 성능 증가하나 항상 그런것은 아니며 학습 시간은 증가함
# max_features : 피쳐 샘플링할 피쳐의 개수, 너무 적으면 성능 저하, 너무 크면 앙상블 효과 저하
# max_samples : 결정트리의 최대 깊이, 결정트리의 과정합 방지, 결정트리의 다른 주요 하이퍼 파리미터도 동일하게 사용 가능
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf_clf = RandomForestClassifier(random_state=156)
rf_clf.fit(X_train, y_train.values.ravel())
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'랜덤 포레스트 정확도: {accuracy:.4f}')
# %% 부스팅 : 여러 개의 약한 학습기를 순차적으로 학습시켜, 이전 모델의 실수를 보완해나가면서 최종적으로 강력한 예측 모델 생성
# 앞선 모델의 틀린 부분에 가중치를 두어 학습 : 데이터셋 -> 모델1 -> 모델1의 결과 -> 모델2 -> 모델2의 결과 -> 모델3 -> 최종 결과값
# 그래디언트 부스팅(GBM) = 이전 모델의 오답의 정도(잔차)를 다음 모델이 계속 학습하고 보완, 잔차를 0에 가깝게 만들어 정답에 근접시킴
# GBM의 하이퍼 파리미터 = n_estimators / learning_rate / max_depth / subsample
# n_estimators : 순차적 학습에 사용할 결정트리의 개수
# learning_rate : 학습률, 이전 모델의 잔차를 다음 모델에 얼마나 강하게 반영할 것인가, n_estimators와 트레이드 오프 관계
# max_depth : 각 결정트리의 최대 깊이, GBM의 경우 3 ~ 5 정도의 얕은 트리 사용이 일반적
# subsample : 트리 학습 시 사용할 훈련 데이터 샘플의 비율, 1보다 작아지면 전체 데이터의 일부만 사용, 과적합 억제 가능
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=156)
gb_clf.fit(X_train, y_train.values.ravel())
end_time = time.time()
y_pred = gb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'GBM 정확도: {accuracy:.4f}')
print(f'GBM 학습 시간: {end_time - start_time:.1f} 초')
# XGBoost = GBM + 병렬처리 + 규제(과적합 억제)
# 사이킷런 래퍼 방식으로 구현 : 사이킷런의 다른 모델들 처럼 패키징, 편의성 및 라이브러리 호환성 높음, 조기중단 편의성 낮음
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
y_train_adj = y_train - 1
y_test_adj = y_test - 1
# XGB의 레이블값은 항상 0부터 시작해야함
start_time = time.time()
xgb_clf = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='mlogloss'
)
xgb_clf.fit(X_train, y_train_adj.values.ravel())
end_time = time.time()
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test_adj, y_pred)
print(f'XGBoost 정확도: {accuracy:.4f}')
print(f'XGBoost 학습 시간: {end_time - start_time:.1f} 초')
# 파이썬 네이티브 방식으로 구현 : 편의성 및 라이브러리 호환성 떨어지나 더 세밀한 제어 및 편리한 조기중단 가능
# 조기중단 = 성능 향상폭이 무의미하면 학습을 일찍 중단, eval_set / early_stopping_rounds / eval_metric 파라미터 필요
# eval_set : 평가용 데이터셋, 학습에 사용되지 않는 검증용 데이터  # eval_metric : 평가 기준
# early_stopping_rounds : 중단 조건, 일정 횟수만큼 최고치를 갱신하지 못 하면 조기중단시킴
import xgboost as xgb
dtrain = xgb.DMatrix(data=X_train, label=y_train_adj)
dtest = xgb.DMatrix(data=X_test, label=y_test_adj)
params = {'max_depth': 3,
          'eta': 0.1, # learning_rate
          'objective': 'multi:softmax',
          'num_class': 6, # 정답 클래스 개수
          'eval_metric': 'mlogloss'
         }
num_rounds = 400 # n_estimators
evals = [(dtrain, 'train'), (dtest, 'eval')]
xgb_native = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds,
                       early_stopping_rounds=50, evals=evals)
# %% LightGBM : 트리의 성장에 있어서 수직적 확장(GBM) 대신 수평적 확장 사용, 수행시간 및 메모리 사용량 개선
# 수평적 확장 : 동일 깊이의 모든 노드를 균형있게 분할, 최적 분할 탐색에 유리하나 리소스와 속도 면에서 비효율적
# 수직적 확장 : 가장 이득이 큰 리프 노드만 집중적으로 분할, 리소스와 속도면에서 매우 효율적이나 과적합 위험성 증가
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import re
import time
def _sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        # numpy 배열 등은 그대로 반환 (feature name 미사용)
        return df

    original_cols = list(df.columns)
    safe_cols = []
    seen = set()

    for col in original_cols:
        # 문자열화 후 안전 문자만 유지(영문/숫자/언더스코어), 나머지는 언더스코어로 치환
        safe = re.sub(r'[^A-Za-z0-9_]', '_', str(col))

        # 빈 문자열이거나 숫자로 시작하면 접두사 부여
        if not safe or safe[0].isdigit():
            safe = f"f_{safe}" if safe else "f_"

        base = safe
        idx = 2
        # 유일성 보장
        while safe in seen:
            safe = f"{base}_{idx}"
            idx += 1

        seen.add(safe)
        safe_cols.append(safe)

    # 새로운 DataFrame으로 반환해 원본 영향 최소화
    df = df.copy()
    df.columns = safe_cols
    return df
X_train_lgb = _sanitize_feature_names(X_train)
X_test_lgb = _sanitize_feature_names(X_test)
# LightGBM은 일부 특수문자는 입력받지 못 함, 허용하지 않는 JSON 특수문자는 제거/치환해야함
start_time = time.time()
lgbm_clf = lgb.LGBMClassifier(
    n_estimators=400, # 트리 400개
    learning_rate=0.1,
    random_state=156
)
evals = [(X_test_lgb, y_test_adj)]
lgbm_clf.fit(X_train_lgb, y_train_adj.values.ravel(),
             eval_set=evals  if 'X_test_lgb' in locals() and 'y_test' in locals() else None,
             eval_metric='logloss',
             callbacks=[lgb.early_stopping(stopping_rounds=50)]) # 50번 동안 성능 향상 없으면 중단
end_time = time.time()
y_pred = lgbm_clf.predict(X_test_lgb)
accuracy = accuracy_score(y_test_adj, y_pred)
print(f'LightGBM 정확도: {accuracy:.4f}')
print(f'LightGBM 학습 시간: {end_time - start_time:.1f} 초')
# %% 스태킹 - 여러 모델(기반 모델)의 학습 결과 자체를 학습 데이터로 삼아 최종 모델(메타 모델)이 한번 더 학습, 예측 성능 가장 우수하나
# 구현이 복잡하고 연산 비용이 가장 큼, 주로 성능 극대화를 위해 모델의 가장 마지막에 적용하는 경우가 많음, 타 기법들과 달리 메타 모델이
# 각 기반 모델들의 예측 성향과 실수를 파악하여, 그 결과를 종합해 가장 가능성이 높은 최종 정답을 도출 따라서 각 기반모델의 단점 보완 가능
# : 데이터셋 -> 기반모델1, 기반모델2, 기반모델3 -> 각 모델의 결과값 -> 메타모델 -> 최종 결과값
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 1. 데이터 로드 및 분할
cancer_data = load_breast_cancer()
X_data = cancer_data.data
y_label = cancer_data.target
X_train , X_test , y_train , y_test = train_test_split(X_data, y_label, test_size=0.2, random_state=0)
# 2. 개별 기반 모델 생성
knn_clf  = KNeighborsClassifier(n_neighbors=4)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
dt_clf = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(n_estimators=100)
# 3. 스태킹 결과를 학습할 최종 메타 모델 생성
lr_final = LogisticRegression()
# 4. 개별 모델 학습
knn_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)
# 5. 개별 모델의 예측 결과를 생성하고 확인
knn_pred = knn_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)
print(f'KNN 정확도: {accuracy_score(y_test, knn_pred):.4f}')
print(f'랜덤 포레스트 정확도: {accuracy_score(y_test, rf_pred):.4f}')
print(f'결정 트리 정확도: {accuracy_score(y_test, dt_pred):.4f}')
print(f'에이다부스트 정확도: {accuracy_score(y_test, ada_pred):.4f}')
# 6. 개별 예측 결과를 하나로 합치기 (행과 열을 바꿈)
# 각 모델의 예측 결과가 [1, 0, 1...] 형태의 1차원 배열이므로, 이를 열 방향으로 합침
stacked_predictions = np.array([knn_pred, rf_pred, dt_pred, ada_pred])
stacked_predictions = np.transpose(stacked_predictions)
# 7. 최종 메타 모델 학습 및 예측
lr_final.fit(stacked_predictions, y_test)
final_pred = lr_final.predict(stacked_predictions)
print(f'최종 스태킹 모델 정확도: {accuracy_score(y_test, final_pred):.4f}')
# %% 교차 검증 스태킹 - 기본 스태킹 기법은 학습 데이터로 예측한 값을 다시 학습에 사용하기 때문에 과적합 발생가능성 높음, 따라서 각 기반 모델
# 학습 시 교차 학습 기법 도입 : 데이터를 K개의 폴드로 나누어 K-1개의 폴드로 학습하고 학습에 사용되지 않은 폴드로 검증하는 것을 K회 반복
# 이때 K회 예측에 따른 결과값을 모두 병합(Concat)하면 전체 학습 데이터와 같은 크기의 '학습 메타 데이터' 생성, 그리고 K회 예측 과정에서
# 생성된 K개의 모델로 전체 테스트 데이터에 대하여 예측 후 K개 예측값을 평균내어 전체 테스트 데이터와 같은 크기의 '테스트 메타 데이터' 생성
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# 개별 모델이 메타 모델을 위한 학습/테스트 데이터를 생성하는 함수
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    # k-fold 교차 검증을 위한 객체 생성
    kf = KFold(n_splits=n_folds, shuffle=False)
    # 메타 모델이 사용할 학습 데이터 저장용 (OOF 예측값)
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    # 메타 모델이 사용할 테스트 데이터 저장용 (평균 예측값)
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__, '모델 학습 시작')
    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        # 학습 데이터에서 폴드별로 학습/검증 데이터 분리
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]
        # 모델 학습
        model.fit(X_tr, y_tr)
        # 1. 검증 데이터로 예측하여 '학습용 메타 데이터' 생성
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
        # 2. 테스트 데이터로 예측하여 나중에 평균낼 값 저장
        test_pred[:, folder_counter] = model.predict(X_test_n)
    # 테스트 데이터 예측값들의 평균을 내어 '테스트용 메타 데이터' 생성
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    return train_fold_pred, test_pred_mean
# 1. 각 기반 모델별로 메타 데이터 생성
knn_train, knn_test = get_stacking_base_datasets(knn_clf, X_train, y_train, X_test, 7)
rf_train, rf_test = get_stacking_base_datasets(rf_clf, X_train, y_train, X_test, 7)
dt_train, dt_test = get_stacking_base_datasets(dt_clf, X_train, y_train, X_test, 7)
ada_train, ada_test = get_stacking_base_datasets(ada_clf, X_train, y_train, X_test, 7)
# 2. 각 모델의 결과물을 옆으로 붙여서 최종 메타 모델용 데이터셋 만들기
Stack_final_X_train = np.concatenate((knn_train, rf_train, dt_train, ada_train), axis=1)
Stack_final_X_test = np.concatenate((knn_test, rf_test, dt_test, ada_test), axis=1)
# 3. 최종 메타 모델(Logistic Regression) 학습 및 정확도 측정
lr_final.fit(Stack_final_X_train, y_train)
stack_final_pred = lr_final.predict(Stack_final_X_test)
print(f'최종 스태킹 정확도: {accuracy_score(y_test, stack_final_pred):.4f}')

# %% GridSearchCV와 베이지안 최적화
# GridSearchCV : 사용자가 테스트하고자 하는 모든 하이퍼파라미터의 조합으로 학습을 시도, K-Fold 교차검증하여 최적 파라미터를 탐색
# 모든 하이퍼 파라미터 조합을 전수조사하므로 하이퍼 파라미터 조합의 크기에 비례해 처리시간 급격히 증가
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 3, 4]
}
grid_search = GridSearchCV(
    estimator=dt_clf_final,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최고 정확도:", grid_search.best_score_)
dt_clf_best = grid_search.best_estimator_
# %% 베이지안 최적화 : 전수조사 대신 이전의 탐색 결과를 바탕으로 다음 탐색지점을 지능적으로 선정, 최소한의 시도로 최적 해답 탐색
# 랜덤한 파라미터를 임의 탐색 - 개략적인 대리 모델 생성 - 획득함수로 다음에 탐색할 파라미터를 선정 - 탐색 후 정보 업데이트 - 반복
# HyperOpt = 파라미터 최적화용 파이썬 라이브러리, 그러드 서치부터 TPE 까지 다양한 알고리즘으로 베이지안 최적화 가능
# HyperOpt의 3요소 = 탐색 공간 / 목적 함수 / fmin() 함수  # fmin() 함수 : 최적화 실행하는 엔진
# 탐색 공간 : 최적화할 하이퍼파라미터의 종류와 범위를 정의, 메소드를 통해 일정범위의 정수 / 일정범위의 실수 / 특정 리스트로 지정 가능
# 목적 함수 : 최소화할 대상이 되는 함수, 성능 평가시 최소화하고자 하는 값(보통 손실값)을 정의, 정확도 사용하고 싶다면 음수로 만들어야
from hyperopt import fmin, tpe, hp
import numpy as np
search_space = hp.uniform('x', -10, 10)
def objective_func(x):
    return (x - 1)**2
best = fmin(
    fn=objective_func,       # 최소화할 목적 함수
    space=search_space,      # 탐색할 하이퍼파라미터 공간
    algo=tpe.suggest,        # 최적화 알고리즘 (TPE 사용)
    max_evals=50,            # 탐색을 시도할 횟수
    rstate=np.random.default_rng(seed=0) # 결과 재현을 위한 시드 설정
)
print(best)
# %% 베이지안 최적화로 XGBoost 튜닝하기
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
xgb_search_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0)
}
def objective_func(search_space):
    # quniform으로 받은 값은 float일 수 있으므로 int로 변환
    xgb_clf = xgb.XGBClassifier(
        n_estimators=int(search_space['n_estimators']),
        learning_rate=search_space['learning_rate'],
        max_depth=int(search_space['max_depth']),
        min_child_weight=int(search_space['min_child_weight']),
        subsample=search_space['subsample'],
        colsample_bytree=search_space['colsample_bytree'],
        eval_metric='mlogloss',
        random_state=156
    )

    scores = cross_val_score(xgb_clf, X_train, y_train_adj, cv=3)
    loss = 1 - np.mean(scores)
    return {'loss': loss, 'params': search_space, 'status': STATUS_OK}
trials = Trials()
best = fmin(
    fn=objective_func,  # 목적 함수
    space=xgb_search_space,  # 탐색 공간
    algo=tpe.suggest,  # 최적화 알고리즘
    max_evals=50,  # 탐색 횟수 (시도 횟수)
    trials=trials,  # 탐색 과정 기록
    rstate=np.random.default_rng(seed=0)
)
print('최적 하이퍼파라미터:', best)

# %% 피쳐 선택 - 불필요하거나 중복된 피처가 많으면 모델이 복잡해져서 과적합이 발생하기 쉽고 학습 속도도 느려짐, 따라서 예측성능을 유의미하게
# 높이는 피쳐들을 선별해야할 필요성 있음 => 필터 방식 / 래퍼 방식 / 임베디드 방식
# 필터 방식 : 피쳐와 타겟 사이의 분산/상관계수 등 통계량만으로 피쳐를 선정, 빠르고 가볍지만 피쳐와 모델간의 상관관계는 고려하지 못함
# 래퍼 방식 : 피쳐 조합을 바꾸어 가며 모델을 여러번 학습 후 가장 성능이 좋은 조합 선택, 성능이 가장 우수하나 매우 무겁고 과적합 위험 존재
# 임베디드 방식 : 모델 자체에 피쳐 선택 기능이 포합되어있음, Lasso 회귀나 결정 트리 등이 임베디드 방식으로 피쳐를 선택함
# Recursive Feature Elimination (RFE) : 래퍼 방식의 일종, 모델에 전체 피쳐로 학습 시킨후 모델이 제공하는 피쳐 중요도나 상관계수 정보
# 를 바탕으로 최하위 피쳐를 제거한 다음 다시 학습, K개의 피쳐가 남을때 까지 해당 과정 반복, 피쳐 간 상호작용이 반영되나 학습 시간이 매우 김
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target
# 기반 모델 생성
rf = RandomForestClassifier(n_estimators=100, random_state=156)
# RFE 객체 생성 (n_features_to_select: 최종적으로 남길 피처 개수)
select = RFE(rf, n_features_to_select=2)
# 학습 및 변환
select.fit(X, y)
X_new = select.transform(X)
print(f"원본 피처 개수: {X.shape[1]}")
print(f"선택된 피처 개수: {X_new.shape[1]}")
# 결정 트리에서 제공하는 Feature Importance의 경우 지니 불순도 감소량(Mean Decrease Impurity, MDI)을 기반으로 하는데
# 이는 고의성 편향(실질적인 중요도가 더 낮더라도 값의 종류가 많은 피처가 선턱될 위험 존재) 및 다중공산성(둘 다 중요한 피쳐라도
# 한쪽이 먼저 선택되면 한쪽의 중요도가 낮다고 간주될 가능성 존재) 뮨제를 일으킴, 주로 아래와 같은 기법을 통해 이 문제를 보완함
# Permutation Importance (순열 중요도) - 의도적으로 특정 피쳐의 데이터를 무작위로 섞은 후 다시 예측, 모델 정확도가 유의미하게 하락 시
# 해당 피쳐의 중요도는 높은 것으로 가정, 이때 해당 피쳐의 분포(평균, 분산 등)는 유지하되 타겟과의 연결 고리만 손상시킴 / SHAP (Shapley
# Additive Explanations) - 게임 이론 기반, 협력 게임에서 각 피처가 예측값에 얼마나 기여했는지를 수학적으로 계산, 모델이 바뀌어도 결과
# 가 일관적이고 데이터의 일부분에 대한 분석도 제공 / Mutual Information (상호 정보량) - 한 피처를 알게 됨으로써 타겟에 대한 불확실성이
# 얼마나 줄어드는지를 계산 / Boruta 알고리즘 - 랜덤 포레스트 기반 알고리즘, 단순 순위가 아닌 통계적 유의성을 바탕으로 판단, 원래 피처들의
# 값을 무작위로 섞어 그림자 피처를 만든 후 원본 피쳐와 함께 학습 시킨 뒤 그림자 피쳐의 최대 중요도보다 중요도가 더 높은 원본 피쳐를 선택 /
# L1 규제 - 선형 모델이나 로지스틱 회귀에서 가중치의 절대값 합을 손실 함수에 추가, 규제 강도가 커질수록 중요하지 않은 피처의 가중치가 감소
# 피쳐 선택은 차원축소와 다름 : 차원축소(원본 피쳐를 조합해 새 피쳐 생성, 낮은 해석력, 정보손실 최소화) vs 피쳐 선택(원본 피쳐의 일부를
# 버리고 일부만 선택, 매우 높은 해석력, 정보손실 발생)