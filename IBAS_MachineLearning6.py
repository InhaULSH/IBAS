# 10. 차원 축소
# 지나치게 많은 차원의 피쳐 -> 차원의 저주 유발 : 각 데이터가 전체 공간에서 차지하는 비율 감소, 데이터가 듬성듬성 존재하는 회소성(Sparse)
# 증가 / 각 데이터 간의 거리가 비슷해지므로 유클리드 거리 기반 계산 무력화 / 데이터는 희소한데 피쳐만 많으면 데이터의 핵심 패턴이 아닌 노이
# 즈를 학습하는 과적합 위험성 급증 / 사람이 데이터 구조를 파악하기 어려워지므로 시각화에 불리해지고 해석 가능성도 급감
# %% 따라서 차원(피쳐)의 수를 줄이되, 데이터의 특성은 보존하여 데이터 속에 숨겨진 핵심 정보를 파악할 수 있도록 해야함
# 피쳐 선택 및 피쳐 추출(원본 피쳐를 저차원으로 투영하여 새로운 피쳐 생성, PCA/LDA/SVD 등)을 이용하여 차원 축소 가능
import matplotlib.pyplot as plt
import numpy as np

# 1차원부터 100차원까지 데이터 포인트 간의 평균 거리 변화 관찰
dimensions = range(1, 101)
avg_distances = []
for d in dimensions:
    # 0과 1 사이의 랜덤한 포인트 1000개 생성
    points = np.random.rand(1000, d)
    # 포인트들 사이의 거리 계산 (유클리드 거리)
    # 차원이 높아질수록 포인트 간의 평균 거리가 멀어짐을 확인
    dist = np.mean(np.linalg.norm(points - 0.5, axis=1))
    avg_distances.append(dist)
plt.figure(figsize=(10, 6))
plt.plot(dimensions, avg_distances, marker='o', color='b')
plt.title('Average Distance from Center as Dimensions Increase')
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Distance')
plt.grid(True)
plt.show()

# %% PCA (주성분 분석) - 피쳐 추출 기법의 일종, 피쳐간의 공분산 행렬(두 변수가 함께 변하는 정도를 나타내는 행렬) 계산 -> 공분산 행렬을
# 분해하여 고유벡터 및 고유값 추출, 이때 고유벡터는 데이터의 분산을 가장 극대화 시키는(데이터를 가장 잘 설명하는) 제1주성분이 되고 고유값
# 은 분산이 됨 -> 주성분들과 직교하면서 주성분을 제외한 나머지 분산을 가장 극대화시키는 벡터를 제N주성분으로 추출히기를 반복 -> 고유벡터
# 를 바탕으로 원래 데이터를 새로운 차원으로 투영
# 이때 PCA는 비지도 학습이며 주성분 추출 시 분산의 절대값을 이용하므로 피쳐간 스케일링은 필수적, PCA의 경우 노이즈 제거, 다중공산성 완화,
# 시각화 용이성 측면에서 유리하지만 해석력 및 정보 손실 측면에서는 다른 기법 대비 다소 불리함
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
# 1. 데이터 로드 (붓꽃 데이터 세트 활용)
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 2. 표준화 (PCA 전 필수 단계)
# 각 피처의 스케일을 맞춰주지 않으면 분산 계산이 왜곡
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df)
# 3. PCA 객체 생성 및 변환 (4차원 데이터를 2차원으로 축소)
# n_components가 0과 1사이의 실수인 경우 전체 변동성(분산)의 일정 비율을 유지하기 위해 필요한 주성분의 개수를 알고리즘이 자동으로 계산
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_scaled)
# 4. 결과 확인
print(f"원본 데이터 형태: {iris_scaled.shape}")
print(f"PCA 변환 후 형태: {iris_pca.shape}")
# 5. 설명된 분산 비율 확인
# 각 주성분이 얼마나 많은 정보를 담고 있는지 출력합니다.
print(f"주성분별 분산 설명 비율: {pca.explained_variance_ratio_}")
print(f"총 설명 가능한 분산 비율: {sum(pca.explained_variance_ratio_):.2f}")
# 6. 시각화
pca_columns = ['pca_component_1', 'pca_component_2']
df_pca = pd.DataFrame(iris_pca, columns=pca_columns)
df_pca['target'] = iris.target
markers = ['^', 's', 'o'] # Setosa: 세모, Versicolor: 네모, Virginica: 동그라미
for i, marker in enumerate(markers):
    x_data = df_pca[df_pca['target'] == i]['pca_component_1']
    y_data = df_pca[df_pca['target'] == i]['pca_component_2']
    plt.scatter(x_data, y_data, marker=marker, label=iris.target_names[i])
plt.legend()
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Iris Data PCA (2 Components)')
plt.show()
# 7. 성능 비교
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_clf = RandomForestClassifier(random_state=42)
# 원본 데이터로 예측 및 평가
rf_clf.fit(X_train_scaled, y_train)
pred_original = rf_clf.predict(X_test_scaled)
acc_original = accuracy_score(y_test, pred_original)
# PCA 후 데이터로 예측 및 평가
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
rf_clf.fit(X_train_pca, y_train)
pred_pca = rf_clf.predict(X_test_pca)
acc_pca = accuracy_score(y_test, pred_pca)
print("=== PCA 전후 예측 성능 비교 ===")
print(f"원본 데이터 정확도: {acc_original:.4f}")
print(f"PCA 데이터 정확도: {acc_pca:.4f}")

# %% LDA(Linear Discriminant Analysis, 선형 판별 분석) - 데이터를 저차원 공간으로 투영했을 때 분할된 클래스 간 분산은 최대화하고,
# 분할된 클래스 내부 분산은 최소화하는 축을 탐색, PCA와 달리 데이터 전체의 분산아닌 클래스간 분별력을 극대화를 목표료하는 지도 학습 방식
# 클래스 정보를 이용하기 때문에 주성분의 개수는 (클래스의 개수 - 1)개 이하여야함, 붓꽃 데이터는 종이 3개이므로 LDA로는 최대 2차원까지만
# 축소 가능, LDA는 차원 축소보다는 분류를 위한 전처리 및 특징 추출에 유리하므로 분류 모델 성능 극대화가 목적이라면 PCA보다 LDA가 더 유리
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
# 1. 데이터 로드 및 표준화
iris = load_iris()
iris_scaled = StandardScaler().fit_transform(iris.data)
# 2. LDA 변환 (클래스 정보를 함께 입력)
# n_components는 클래스 개수(3) - 1인 2로 설정
lda = LinearDiscriminantAnalysis(n_components=2)
# fit 호출 시 지도학습이므로 iris.target을 반드시 넣어주어야 함
lda.fit(iris_scaled, iris.target)
iris_lda = lda.transform(iris_scaled)
# 3. 시각화 준비
lda_columns = ['lda_component_1', 'lda_component_2']
df_lda = pd.DataFrame(iris_lda, columns=lda_columns)
df_lda['target'] = iris.target
# 4. 결과 시각화
markers = ['^', 's', 'o']
for i, marker in enumerate(markers):
    x_data = df_lda[df_lda['target'] == i]['lda_component_1']
    y_data = df_lda[df_lda['target'] == i]['lda_component_2']
    plt.scatter(x_data, y_data, marker=marker, label=iris.target_names[i])
plt.legend()
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('Iris Data LDA (2 Components)')
plt.show()
# 5. 성능 비교
X_train, X_test, y_train, y_test = train_test_split(
    iris_scaled, iris.target, test_size=0.3, random_state=156
)
rf_clf_org = RandomForestClassifier(random_state=156)
# 원본 데이터로 예측 및 평가
rf_clf_org.fit(X_train, y_train)
pred_org = rf_clf_org.predict(X_test)
accuracy_org = accuracy_score(y_test, pred_org)
# LDA 후 데이터로 예측 및 평가
iris_lda_train = lda.transform(X_train)
iris_lda_test = lda.transform(X_test)
rf_clf_lda = RandomForestClassifier(random_state=156)
rf_clf_lda.fit(iris_lda_train, y_train)
pred_lda = rf_clf_lda.predict(iris_lda_test)
accuracy_lda = accuracy_score(y_test, pred_lda)
print(f'원본 데이터(4개 피처) 정확도: {accuracy_org:.4f}')
print(f'LDA 변환 데이터(2개 피처) 정확도: {accuracy_lda:.4f}')

# %% SVD(Singular Value Decomposition, 특이값 분해) - 임의의 행렬 A를 다음과 같이 분해함 : A = U * Sigma * V^T => 이때 U와 V^T
# 는 A*A^T를 고유값 분해해서 얻은 고유벡터들로 구성, Sigma는 대각 성분에 A의 특이값이 큰 순서대로 나열되어 있음, 특이값은 데이터의 에너지
# 또는 중요도를 나타내며, 0이 아닌 특이값의 개수가 행렬의 차원과 같음
# 이때 행렬의 대각 성분 중 상위 K개만 남기고 나머지는 버리는 Truncated SVD를 이용하면 데이터의 양은 크게 줄어들면서도 데이터의 특징은 보존
# 가능, LSA(단어들 사이의 잠재적인 의미를 파악하는 텍스트 마이닝 기법) / 추천 시스템(사용자-아이템 평점 행렬에서 사용자의 취향 추출에 사용)
# / 이미지 압축(이미지 데이터에서 작은 특이값들을 제거함으로써 화질 저하를 최소화하며 용량 축소) / PCA(데이터의 평균을 0으로 맞춘 상태에서
# SVD를 수행하면 PCA와 동일) 등에 널리 활용됨
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# 1. 데이터 로드
iris = load_iris()
iris_ftrs = iris.data
# 2. Truncated SVD 변환 (2개 컴포넌트로 압축)
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_ftrs)
iris_tsvd = tsvd.transform(iris_ftrs)
# 3. 시각화
plt.scatter(x=iris_tsvd[:,0], y=iris_tsvd[:,1], c=iris.target)
plt.xlabel('TruncatedSVD Component 1')
plt.ylabel('TruncatedSVD Component 2')
plt.title('Iris Data by Truncated SVD')
plt.show()
# %% NMF(Non-Negative Matrix Factorization) - SVD와 유사하나 행렬 내의 모든 원소가 양수인 경우에만 사용 가능, 임의의 행렬 V를 다음
# 과 같이 분해 V ≒ W(가중치) × H(특징), 모든 원소가 양수이기 때문에, 특징들을 더해서 전체 데이터를 구성하는 방식으로 해석 가능
# SVD 보다 해석이 용이하므로 이미지 처리('부분적 특징'들을 추출에 사용, SVD는 음수가 포함되어 해당 작업 불가) / 텍스트 마이닝 (문서군에서
# 특정 단어들의 조합 탐색에 사용) 등에 널리 활용됨
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# 1. 데이터 로드
iris = load_iris()
iris_ftrs = iris.data
# 2. NMF 변환 (2개 컴포넌트)
nmf = NMF(n_components=2, max_iter=500)
nmf.fit(iris_ftrs)
iris_nmf = nmf.transform(iris_ftrs)
# 3. 시각화
plt.scatter(x=iris_nmf[:,0], y=iris_nmf[:,1], c=iris.target)
plt.xlabel('NMF Component 1')
plt.ylabel('NMF Component 2')
plt.title('Iris Data by NMF')
plt.show()