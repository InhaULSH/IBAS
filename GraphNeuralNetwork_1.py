# 그래프 데이터
# 테이블형 데이터에서는 한 행이 곧 한 관측단위이며 한 열이 곧 한가지 속성, 직관적이고 전통적인 분류/회귀 모델에 바로 적용 가능, 그러나 각 관
# 측단위간 복잡하고 계층적인 관계가 존재하는 데이터를 표현하기에는 부족함 : 금융 분야의 사기 탐지 시스템에서 각 개별 사례의 피쳐를 들여다보는
# 것보다는 이 거래가 다른 거래와 어떤 연관성이 있는가를 파악하는 것이 더 효과적, 이커머스 분야에서 같은 상품을 여러번 본 사용자에게 해당 상품
# 을 다시 보여주거나 비슷한 상품을 많이 본 사람들의 행동을 통계적으로 합쳐서 추천하기보다는 어떤 상품들이 실제로 서로 함께 소비되는지 또는 어
# 떤 사용자 집단이 서로 비슷한 취향의 네트워크를 이루는지와 같은 구조적 관계를 파악하는 것이 더 효과적 => 노드와 엣지 개념 도입

# 그래프 데이터는 각 관측단위는 노드로, 각 속성은 엣지로 취급, 거대한 네트워크 형태의 그래프로 표현 가능, 의심되는 패턴은 그래프의 구조로 확
# 인 가능 : 금융 분야의 사기 탐지 시스템에서 한 디바이스에서 여러 계정이 번갈아 로그인하고 서로 다른 카드로 여러 가맹점에서 결제한다면 이 디
# 바이스는 일종의 허브처럼 보일 것, 이커머스 분야에서 사용자/상품/셀러/카테고리를 노드로 나타내고 구매 시 서로간의 엣지를 생성함으로서 각 관
# 측단위간 다양한 상호 작용을 자연스럽게 확인할 수 있을 것 => GNN은 이러한 네트워크 구조를 이용해 이웃 노드의 정보와 자기자신 노드의 정보를
# 바탕으로 구조와 속성을 모두 활용하는 임베딩 생성, 여기서 임베딩이란 어떤 객체를 연속적인 수치 벡터로 나타내되 그 의미와 관계를 반영해 거리
# 로 표현한 것을 말함(단어를 예로 들면 비슷한 단어끼리는 유사한 수치로 표현하고 비슷하지 않은 단어 끼리는 차이가 큰 수치로 표현한 벡터)

# 기존 테이블 기반 모델을 유지하되 모델의 성능이 정체된 지점에서 그래프 정보를 추가적인 신호로 활용할 수 있는지 탐색한 뒤, 그래프 스키마를 설
# 계해서 주요 엔티티와 관계를 정리하고 이를 토대로 GNN으로 임베딩을 학습한 다음, GNN에서 나온 임베딩을 기존 모델의 피처로 넣어 성능 향상 정
# 도를 확인하고, 그래프 구성/샘플링 전략/학습 비용/배포 복잡도를 고려해 어디까지 GNN을 쓰고 어디까지는 기존 모델/규칙 기반을 유지할지 합리적
# 인 타협점을 설정하는 과정이 중요 : 기존 테이블 기반 모델이 잘 작동한 다면 굳이 복잡한 GNN을 사용팔 필요는 없기 때문


# %% GNN(Graph Neural Network)
# 뉴럴 네트워크(딥러닝)의 대부분 모델은 규칙적인 구조를 가진 데이터를 가정, 그러나 금융 네트워크나 이커머스 사용자–상품 관계는 미리 정해진 규
# 칙이 없고 노드마다 연결 수가 제각각이며 구조가 수시로 변함, 이를 무리하게 CNN/LSTM에 사용 시 정보 손실이 심화되거나 차원의 저주 및 극단적
# 인 희소행렬 발생할 가능성 존재, 이를 GNN의 메세지 패싱 개념을 통해 처리 가능 : 각 노드는 처음에 자신 고유의 속성 벡터를 갖고 시작(금융 데
# 이터의 경우 거래 금액/거래 시간/가맹점 종류 등) -> 각 노드는 이웃 노드들로부터 정보를 모은 뒤(이웃들의 임베딩을 평균 내거나 가중합을 구함)
# -> 모아진 이웃 정보와 자기 자신의 정보를 조합해서 새로운 임베딩을 생성(행렬 곱과 ReLU 등 비선형 활성함수를 사용) -> 이 과정을 여러번 반복
# 하여 보다 먼 거리의 이웃노드들의 정보도 자기자신의 임베딩에 점차 반영

# 기존 그래프 기반 기법은 그래프의 구조적으로 비슷한 위치에 있는 노드를 비슷한 임베딩으로 변환하는 과정과 해당 임베딩을 모델에 입력해 학습시키
# 는 과정이 분리되어있음, 각 임베딩은 우리가 관심 있는 목표를 반영하지 않고 그래프의 구조만 반영하므로, 비슷한 연결 패턴을 가졌지만 하나는 사
# 기 하나는 정상인 경우과 같은 패턴을 제대로 학습하지 못 함, 반면 GNN은 메세지 패싱을 통한 임베딩 생성 - 임베딩 기반 모델 학습 및 예측 - 역
# 전파를 통해 출력의 손실값을 임베딩 생성 과정 전체로 전달 이라는 과정이 연속적으로 진행되므로 원하는 결과에 가까워지도록 임베딩이 조금씩 수정
# 됨, 최종적으로 임베딩을 구체적인 비즈니스 목적에 최적화된 벡터로 만들 수 있음 -> 모델이 특정 정보의 유용성이나 특정 경로의 위험성과 같은 사
# 람이 인지하지 못 하는 수준의 패턴을 학습하고 이를 필요한 만큼만 적절한 수준으로 반영되도록 할 수 있음

# GNN을 구현하는 기법에는 다음과 같은 종류가 존재함 : GCN = 가장 간단한 기법 중 하나이며, 레이어마다 이웃노드의 정보에 대해 대표값(평균 등)
# 을 집계하여 벡터를 만들고 해당 벡터와 자기 자신 노드의 정보를 결합하여 임베딩 생성, 레이어의 단계가 올라갈수록 수집되는 이웃의 범위가 증가,
# 최종 생성된 임베딩을 기존 모델에 피쳐로 추가한 후 성능 개선 정도를 평가하는 방법으로 사용, 그래프 전체를 처리해야 하므로 비실시간 응답을 수
# 백만 노드 이하의 그래프로 생성 할때 사용 가능 / GraphSAGE = GCN과 달리 그래프에서 균등하게 추출한 일부분인 미니 배치만을 GNN에 사용, 이
# 때 업데이트하고자 하는 노드에 대해 이웃 노드 중 일정 개수를 무작위 추출하여 이웃 노드의 정보를 집계하고 이 과정을 이웃의 범위를 늘려가며 반
# 복, 따라서 계산하는 총 이웃 수의 상한이 정해져있어 계산 비용을 억제할 수 있고 새로운 노드에 대해 Inductive한 학습이 가능해짐, 그래프의 노
# 드가 수천만 노드 이상이거나 실시간 응답 생성이 필요하거나 콜드스타트 문제를 해결할 필요가 있을때 사용 가능 / GAT = 평균 또는 졍규화된 합을
# 기준으로 집계해 각 노드를 동등하게 대우하는 앞선 기법들과 달리, 기준 노드와 이웃 노드의 임베딩 또는 피쳐를 비교해 관계의 중요도에 따라 점수
# 를 다르게 부여한 다음 이를 정규화해서 가중치를 부여, 해당 가중치로 각 이웃 노드를 가중 합하여 벡터를 정의, 노드간의 관계가 복잡하고 각 노드
# 간 중요도 편차가 존재하는 대부분의 금융/이커머스 데이터에 유용하게 사용 가능하나 리소스 사용량이나 구현 난이도는 가장 높음 ==> 성능/리소스/
# 복잡도간 트레이드 오프 관계를 조율하여 모델을 사용해야함

# 어떤 GNN을 쓸지보다 어떻게 GNN을 현재 비즈니스 환경에 적용할지가 중요하므로 파이프라인 설계 과정을 거치며 필요한 요소를 점검할 필요 있음 :
# 원천 데이터 수집과 도메인 이해 : 데이터를 적절히 수집한 다음 데이터의 ID가 일관되게 관리되었는지 시계열 정보가 충분히 남아있는 지 등을 확인
# 하고 도메인 지식을 바탕으로 어떤 관계가 이번 모델의 핵심인지를 정의해야함 => 그래프 스키마 설계 : 어떤 엔티티를 노드로 어떤 엔티티를 엣지로
# 볼 것 인지 정의해야함, 노드의 종류가 높아질수록 표현력을 좋아지나 구현이 어려워지고 모델의 전반적인 복잡도도 증가하므로 적은 종류로 시작해서
# 점차 늘려가며 성능을 확인 => 피처 엔지니어링 : 노드와 엣지에 엔티티의 어떤 피쳐를 붙일 것인 가를 결정해야함, 피쳐가 많이 사용될수록 정보량
# 이 늘어나지만 전반적인 복잡도 및 전처리 난이도가 증가함, 범주형 데이터는 원핫인코딩 대신 임베딩 레이어로 처리하는 것이 적절, 스케일이 큰 수
# 치형 피쳐에 대해서 정규화 및 스케일링이 필요함 => 학습/검증/테스트 분할 : 금융/이커머스 데이터에는 시계열이 존재하므로 데이터를 무작위로 분
# 할하면 정보 누수가 발생할 가능성 존재, 따라서 시간 기준으로 특정 시점 이전의 데이터는 훈련에 이후 데이터는 검증/테스트에 사용하는 것이 적절
# => 모델 학습 : 적절한 기법을 선택하여 학습 진행 => 배포와 운영 : GNN을 어디까지 실시간 경로에 넣을 것인지 재학습/재배포 주기를 어떻게 잡
# 을 것인지 등 어떻게 현업 시스템에 붙일지를 결정해야함


# %% 금융 이상 거래 탐지와 GNN
# 기존에는 금융 이상 거래 탐지를 위해 각 거래를 한 줄의 레코드로 보고 금액/시간/국가/가맹점코드/채널 등 피쳐를 바탕으로 분류를 실시했음, 그러
# 나 앞서 설명했듯이 네트워크형 범죄로 진화하는 금융 범죄를 탐지하기 부적절해짐, 누가 누구와 어떻게 연결되어 있는지를 파악하여 분류해야할 필요
# 가 있음 따라서 GNN 기법 도입의 필요성 증가 : 금융 데이터는 크게 고객 및 계정 계층/거래 및 채널 계층/수취인 및 가맹점 계층/기타 정보 로 분
# 류할 수 있음 => 이때 모델의 스키마를 어떻게 정의하고 노드와 엣지에 어떻게 정보를 분배할 지 결정해야함 - 단순형 스키마(거래를 노드로 거래들
# 끼리의 관계를 엣지로 표현, 이때 두 거래가 같은 계정/카드/디바이스/가맹점 등으로 연결된 경우 엣지로 연결, 구현이 단순하고 유지보수가 용이하
# 나 거래 주체에 대한 정보가 간접적으로 표현될 수밖에 없고 피쳐의 활용성 및 복잡한 상황에 대한 표현력이 떨어지게됨) vs 이종 스키마(거래의 주
# 체를 노드로 두고 주체간의 거래가 발생하면 엣지로 연결, 따라서 관계 패턴이 더 명확하게 표현되고 각 주체에 대해 정보를 충분히 표현할 수 있으
# 나 구현 난이도 및 유지보수 비용/연산 비용 등이 증가함) 이때 데이터를 모델 스키마에 맞게 노드 및 엣지로 정보를 분배해야함 => 이러한 모델 스
# 키마를 바탕으로 그래프 궂로 데이터를 구성하면 거래 또는 거래 주체간 연결구조를 GNN으로 표현 가능 => 각 노드/엣지/서브 그래프에 대해서 이상
# 그래프 의심 여부를 판단할 수 있음 => 다만 금융기관은 성숙된 전통적인 이상탐지 모델을 이미 보유하고 있으므로 GNN으로 모델을 교체하기에는 리
# 스크가 지나치게 클 위험있음 따라서 실무적으로는 그래프 정보를 반영한 고급 임베딩을 만드는 피처 팩토리 역할로 사용하는 것이 현재 비즈니스 환
# 경에 더 적합함

# 다만 모델의 스키마와 비즈니스 환경 적용 방향을 결정했더라도 금융 데이터에서는 다음 요소를 추가로 고려해야함 : 라벨링 문제 = 이상 거래 탐지
# 라벨의 경우 현실에서는 극단적으로 불균형하고 이상 거래가 발생한 시점과 그것이 사기로 확정되는 시점 사이에는 시간차가 존재하며 실제로는 사기
# 인데 아직 발견되지 않은 케이스가 많은 문제점이 존재, 따라서 학습에 사용가능한 데이터의 범위가 크게 제한됨, 사기 클래스에 더 큰 가중치를 주
# 거나(손실 함수 가중치), 미니 배치 구성 시 사기 노드는 일정 비율 이상 포함하고 정상 노드는 다운 샘플링 하거나, 정상 샘플보다 어려운/희귀한
# 사기 샘플에 더 큰 페널티를 주는 손실 함수를 사용하는 등의 방법으로 이 문제를 해결할 수 있음 또한 평가 지표 선택 시 정확도는 의미가 없으므
# 로 Precision/Recall/F1/ROC-AUC 특히 PR-AUC를 적극적으로 활용해야함 / 시계열과 스냅샷 = 금융 트랜잭션은 본질적으로 시계열 데이터이므
# 로 GNN에서도 시계열을 어떻게 접근할지 결정해야함, 스냅샷 기반 정적 그래프(특정 기간의 거래를 묶어 그래프 하나를 만들고 해당 기간의 구조를
# 기반으로 사기 여부를 예측하는 모델을 학습, 이후 새로운 기간에 대해 그래프를 재구성하고 같은 모델/파이프라인을 적용, 구현이 비교적 단순하고
# 1차 실험에 적합) vs 시계열 그래프(그래프가 시간에 따라 진화하는 것으로 보고 시간축을 따라 메시지 패싱 하거나 여러 스냅샷을 연속 처리하는
# Temporal GNN을 사용, 어느 기간에 어떤 경로로 자금이 흘렀는지가 중요하므로 실무에서 점점 확대되고 있는 접근)


# %% GNN 기반 금융 이상 거래 탐지 실습 - Elliptic 데이터셋 기반
# 약 200K 비트코인 트랜잭션 노드 + 트랜잭션간 비트코인 자금 흐름을 나타내는 방향 엣지 +  트랜잭션마다 160여 개의 수치형 피쳐 + 49개의 시계
# 열 스텝으로 구성된 데이터셋, 해당 데이터셋을 바탕으로 단일 타입 노드 기반 GNN을 구현하여 불법 트랜잭션을 식별하는 노드 분류 문제를 해결하는
# 것을 목표로 설정 => 이때 GNN은 성능 측면에서 단일 트랜잭션 피처만으로는 잡기 어려운 패턴을 주변 그래프 구조를 통해 포착할 수 있도록하는 역
# 할을, 설명 가능성측면에서 어느 정도 이상 탐지 후 왜 이 거래가 위험해 보이는지를 네트워크 관점에서 분석할 수 있도록 하는 역할을 함

# 데이터 전처리와 GNN 입력 준비 : 해당 데이터셋은 이미 노드와 엣지에 정보가 적절히 분배되어있고 각 CSV 파일에 필요한 정보가 정리되어있음 그
# 러나 이러한 경우에도 GNN이 잘 학습할 수 있는 형태로 데이터를 잘 설계할 필요성 존재 => 노드 인덱스 정합성(GNN 라이브러리는 보통 0~N-1 범
# 위의 연속 정수 인덱스를 기대, 따라서 트랜잭션 ID를 내부용 인덱스로 매핑해야 하며  엣지 리스트와 피처 테이블 전체에 일관되게 적용어야함) /
# 피처 행렬 구성(Elliptic 피처 CSV에는 각 트랜잭션의 Time Step 및 다수의 수치형 피처가 들어 있는데 이중 어떤 피쳐를 노드 피쳐로 사용할지
# 결정해야함, 보통 Time Step을 제외한 나머지 수치형 피처를 모두 사용, 특별히 이상한 값이나 상수 피처나 지나치게 희소한 피처가 있다면 제거를
# 고려할 수 있으나 전체 수치형 피쳐 사용하여 학습 후 정제하는 것이 더 효과적) / 라벨과 마스크 구성(Elliptic에는 세 가지 라벨이 있는데 Unk
# nown은 아직 불법성 여부를 모르는 거래에 해당하며, 이러한 데이터는 지도 학습에서 보통 라벨 대상으로 쓰지 않으므로 학습/검증/평가 대상에서는
# 제외해야함, 단 이 노드들이 이웃으로서 간접적인 역할을 수행할수 있으므로 노드 자체를 제거해서는 안 됨) / 시계열 기반 데이터셋 분할(Ellipti
# c에는 Time Step이 총 49개 존재, 각 스텝은 비트코인 블록체인 상의 특정 시간 구간을 나타낸다고 볼 수 있고 과거로 학습해서 미래를 예측해야
# 하므로 초기 일부 스텝은 Train으로 중간 일부 스텝은 Valid로 마지막 일부 스텝은 Test로 분배함이 적절)

# %% Elliptic CSV를 PyTorch Geometric 데이터로 변환
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# 라이브러리 준비
DATA_DIR = "DataSet_DeepLearning/Elliptic"
features_path = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
edges_path = os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv")
classes_path = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")
features_df = pd.read_csv(features_path, header=None)
edges_df = pd.read_csv(edges_path)
classes_df = pd.read_csv(classes_path)
print(features_df.head())
print(edges_df.head())
print(classes_df.head())
# 데이터 로드

# features_df: 첫 열 = txId, 두 번째 열 = time_step, 나머지 = feature_0 ...
num_cols = features_df.shape[1]
col_names = ['txId', 'time_step'] + [f'feat_{i}' for i in range(num_cols - 2)]
features_df.columns = col_names
# classes_df: txId, class
classes_df.columns = ['txId', 'class']
classes_df['class'] = classes_df['class'].replace('unknown', -1).astype(int)
# edges_df: txId_from, txId_to
edges_df.columns = ['txId_from', 'txId_to']
print(features_df.head())
print(classes_df['class'].value_counts())
# Elliptic의 피처 CSV는 열 이름이 없이 제공되므로 열이름 수동 지정

# 모든 txId 수집
all_txIds = features_df['txId'].values
# 정렬해 두면 나중에 디버깅할 때 편함
all_txIds_sorted = np.sort(all_txIds)
# txId -> index (0 ~ N-1) 매핑 딕셔너리
txid_to_idx = {txId: idx for idx, txId in enumerate(all_txIds_sorted)}
num_nodes = len(all_txIds_sorted)
print("Number of nodes (transactions):", num_nodes)
# 트랜잭션 ID를 내부 인덱스 매핑
# 이때 edges_df와 classes_df에 있는 txId들도 이 매핑을 통해 index로 바꾸어야함

# features_df를 txId 기준으로 정렬하여, all_txIds_sorted 순서와 맞춘다.
features_df_sorted = features_df.set_index('txId').loc[all_txIds_sorted].reset_index()
# time_step과 피처를 분리
time_steps = features_df_sorted['time_step'].values
feature_cols = [c for c in features_df_sorted.columns if c.startswith('feat_')]
features_df_sorted[feature_cols] = features_df_sorted[feature_cols].replace([np.inf, -np.inf], np.nan)
features_df_sorted[feature_cols] = features_df_sorted[feature_cols].fillna(
    features_df_sorted[feature_cols].median(numeric_only=True)
).fillna(0)
x = torch.tensor(features_df_sorted[feature_cols].values, dtype=torch.float)
print("Feature matrix shape:", x.shape)  # [num_nodes, num_features]
# 노드별 피처 벡터 만들기 : 노드0 [0.3,1.2,−0.7,...] / 노드1 [−0.1,0.5,0.9,...] / 노드2...
# Elliptic 피처는 이미 정규화된 수치형 벡터에 가깝게 제공되므로 별도 스케일링 없이 그대로 사용

src = edges_df['txId_from'].map(txid_to_idx)
dst = edges_df['txId_to'].map(txid_to_idx)
mask = src.notna() & dst.notna()
if int((~mask).sum()) > 0:
    print("Dropped transaction edges with unknown endpoints:", int((~mask).sum()))
edge_array = np.vstack([
    src[mask].astype(np.int64).values,
    dst[mask].astype(np.int64).values,
])
edge_index = torch.tensor(edge_array, dtype=torch.long)
assert int(edge_index.max()) < num_nodes, "transaction edge index out of range"
print("Edge index shape:", edge_index.shape)  # [2, num_edges]
# 양방향 그래프의 경우
edge_index_undirected = torch.cat(
    [edge_index, torch.flip(edge_index, dims=[0])],
    dim=1
)
# 엣지 리스트를 edge_index 행렬로 변환 : edges_df의 txId_from, txId_to를 내부 인덱스로 변환하고 PyG에서 기대하는 형태인 [2, num_edg
# es] 크기의 edge_index 행렬 생성, 매핑되지 않는 endpoint가 있으면 해당 엣지만 제거

# txId 기준으로 라벨을 정렬
classes_df_indexed = classes_df.set_index('txId')
# 기본값은 -1 (unknown / unlabeled)
y = torch.full((num_nodes,), -1, dtype=torch.long)
label_map = {1: 0, 2: 1}  # 1=licit -> 0, 2=illicit -> 1
for i, txId in enumerate(all_txIds_sorted):
    if txId in classes_df_indexed.index:
        cls_val = int(classes_df_indexed.loc[txId, 'class'])
        if cls_val in label_map:
            y[i] = label_map[cls_val]
        # cls_val == -1 (unknown)은 그대로 -1 유지
# labeled 노드 마스크
labeled_mask = y >= 0
print("Total labeled nodes:", labeled_mask.sum().item())
print("Class distribution (0=licit,1=illicit, -1=unknown):", torch.bincount(y + 1))
# 각 라벨을 정수로 바꾸고 unknown에 해당하는 노드를 마스크로 처리

time_steps_tensor = torch.tensor(time_steps, dtype=torch.long)
train_mask = (time_steps_tensor >= 1) & (time_steps_tensor <= 34) & labeled_mask
val_mask   = (time_steps_tensor >= 35) & (time_steps_tensor <= 42) & labeled_mask
test_mask  = (time_steps_tensor >= 43) & (time_steps_tensor <= 49) & labeled_mask
print("Train nodes:", train_mask.sum().item())
print("Val nodes:", val_mask.sum().item())
print("Test nodes:", test_mask.sum().item())
# 시계열 기반 Train/Valid/Test Mask 만들기

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data.time_step = time_steps_tensor
data.time_step_mask = F.one_hot(time_steps_tensor - 1, num_classes=49).to(torch.bool)
print(data)
# PyG Data 객체로 묶기

# %% Elliptic 그래프에서 GCN으로 불법 트랜잭션 분류
# 노드 자체의 피처와 그 노드가 어떤 다른 트랜잭션들과 어떻게 연결되어 있는지를 바탕으로 라벨이 없는 새 트랜잭션이 들어왔을 때 illicit일 확률
# 을 잘 예측하는 모델 설계가 목적, 따라서 출력은 각 노드가 illicit일 확률이며 손실은 licit/illicit 라벨이 있는 노드들에 대한 Cross-Entr
# opy가 되고 역전파를 통해 이 손실을 줄이는 방향으로 GCN의 가중치가 업데이트됨, 이를 반복하면 모델은 불법 트랜잭션 주변에 보이는 전형적인 패
# 턴과 정상 트랜잭션 주변의 패턴을 구분하도록 GCN 레이어의 필터가 조금씩 조정됨
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
# 라이브러리 준비
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        # 첫 번째 GCN 레이어: 입력 피처 -> 숨은 차원
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 두 번째 GCN 레이어: 숨은 차원 -> 출력 차원(2 클래스)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 1층: 이웃 정보 집계 + 비선형 활성화
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 2층: 다시 이웃 정보 집계 후, 선형 출력 (logits)
        x = self.conv2(x, edge_index)
        return x
# 2-layer GCN 분류기 정의
# 노드 피처 행렬 X와 edge_index를 받아 두 번의 메시지 패싱을 통해 새로운 노드 표현(그리고 클래스 로짓)을 만들게 됨, conv1에서 각 트랜잭션
# 노드가 1-hop 이웃의 피처를 받아들여 1차 임베딩을 만들고, conv2에서 1차 임베딩을 가지고 다시 한번 이웃 정보를 통합해 최종적인 2-hop 기반
# 표현을 만듦, 출력 차원 out_channels는 클래스 수(2가지)

# Elliptic 데이터에서는 illicit 비율이 극단적으로 낮은 불균형 데이터임, 이 상태에서 그냥 일반 cross-entropy를 쓰면 모델이 거의 다 lici
# t이라고 예측하면서도 높은 Accuracy를 얻는 상태로 빠지기 쉬움, 따라서 사기(illicit) 클래스에 더 큰 가중치를 줘서 놓쳤을 때의 페널티를 키
# 우고, 평가 지표로 Accuracy 대신 ROC-AUC/PR-AUC를 사용하도록 하여 해당 문제를 억제해야함
# 데이터 객체를 device로 올리기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
# 라벨이 있는 노드들만 대상으로 클래스 비율 계산
# 예: tensor([num_licit, num_illicit])
y_labeled = data.y[data.train_mask]
class_counts = torch.bincount(y_labeled)
# 역비율로 가중치 설정 (간단한 예)
class_weights = class_counts.float().sum() / (2.0 * class_counts.float())
# 따라서 CrossEntropyLoss는 [num_classes] 형태의 weight를 받음
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
model = GCNNet(
    in_channels=data.num_features,
    hidden_channels=64,
    out_channels=2,
    dropout=0.5
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # [num_nodes, 2]
    # train 노드만 선택
    train_mask = data.train_mask
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
# 학습 루프 정의 : forward 후 나온 logits에서 train_mask가 True인 노드만 골라 loss 계산, val/test에서도 해당 마스크로 필터링해 평가
@torch.no_grad()
def evaluate(model, data, mask_name):
    model.eval()
    out = model(data.x, data.edge_index)
    mask = getattr(data, mask_name)  # 'val_mask' or 'test_mask'
    logits = out[mask]
    y_true = data.y[mask]
    # 소프트맥스로 확률로 변환
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # illicit 클래스 확률
    y_true_np = y_true.detach().cpu().numpy()
    # 일부 시점에는 mask에 아무 라벨도 없을 수 있으므로 예외 처리
    if len(np.unique(y_true_np)) < 2:
        return None, None
    roc = roc_auc_score(y_true_np, probs)
    pr  = average_precision_score(y_true_np, probs)  # PR-AUC
    return roc, pr
# ROC-AUC와 PR-AUC 계산
num_epochs = 50
best_val_pr = 0.0
best_state = None
for epoch in range(1, num_epochs + 1):
    loss = train_one_epoch(model, data, optimizer, criterion)
    val_roc, val_pr = evaluate(model, data, 'val_mask')
    if val_pr is not None and val_pr > best_val_pr:
        best_val_pr = val_pr
        best_state = model.state_dict()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val ROC: {val_roc:.4f} | Val PR: {val_pr:.4f}")
# 전체 학습 루프 실행
if best_state is not None:
    model.load_state_dict(best_state)
test_roc, test_pr = evaluate(model, data, 'test_mask')
print(f"Test ROC-AUC: {test_roc:.4f} | Test PR-AUC: {test_pr:.4f}")
# 최적 모델 로드 : Test PR-AUC가 가장 높은 모델 로드

# %% GraphSAGE로 확장성 고려하기
# 실제 카드사/거래소/핀테크 환경처럼 그래프가 수천만 노드 수억 엣지 수준이 되면 GCN으로 처리하기 불가능해짐, 따라서 GraphSAGE을 통해 샘플링
# 을 기반으로한 확장 가능 GNN 및 새로운 노드에도 바로 쓸 수 있는 인덕티브 GNN의 구현이 가능해짐, GraphSAGE은 그래프를 여러 미니배치 서브그
# 래프로 나누고 각 미니 배치에서 선택된 일부 시드 노드들에 대해서만 정보를 업데이트, 업데이트 시 각 노드에 대해 이웃 중에서 k개만 무작위로 샘
# 플링해서 집계, 따라서 forward/backward에서 처리해야 하는 이웃 수가 k로 제한되므로 그래프 전체 크기와 관계없이 계산량이 안정적으로 유지됨,
# 또한 Inductive Learning이 가능해 이웃의 피처와 구조만 알고 있으면 학습된 파라미터를 새 노드에도 그대로 적용 가능함
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
# 라이브러리 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GraphSAGE(
    in_channels=data.num_features,
    hidden_channels=64,
    num_layers=2,
    out_channels=2,   # licit vs illicit
    dropout=0.5
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# GraphSAGE 모델 정의

# 라벨 있는 노드들만 seed 후보로 사용
labeled_mask = data.y >= 0
train_idx = torch.where(data.train_mask & labeled_mask)[0]
val_idx   = torch.where(data.val_mask   & labeled_mask)[0]
test_idx  = torch.where(data.test_mask  & labeled_mask)[0]
print("Train seeds:", train_idx.size(0))
print("Val seeds:", val_idx.size(0))
print("Test seeds:", test_idx.size(0))
# Train/Valid/Test용 서브그래프용 미니배치 구성

# 한 번에 몇 개의 seed 노드를 가져올지
batch_size = 1024
# 각 GNN 레이어마다 몇 개의 이웃을 샘플링할지
num_neighbors = [25, 10]  # 2-layer GraphSAGE 기준 예시
train_loader = NeighborLoader(
    data,
    input_nodes=train_idx,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True
)
val_loader = NeighborLoader(
    data,
    input_nodes=val_idx,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=False
)
test_loader = NeighborLoader(
    data,
    input_nodes=test_idx,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=False
)
# Seed 후보들을 기반으로 Neighbor Sampling을 사용하여 Train/Valid/Test용 서브 그래프 구성

def train_sage_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # batch.x, batch.edge_index, batch.y는 서브그래프 기준이다.
        out = model(batch.x, batch.edge_index)  # [num_batch_nodes, 2]
        # 이 중 맨 앞 batch.batch_size개가 이번 미니배치의 seed 노드들이다.
        # 이 노드들에 대해서만 loss를 계산한다.
        out_seed = out[:batch.batch_size]
        y_seed = batch.y[:batch.batch_size]
        # unknown(-1) 라벨 제거 (이론상 여기엔 없어야 하지만 안전하게)
        mask_seed = y_seed >= 0
        if mask_seed.sum() == 0:
            continue
        loss = F.cross_entropy(out_seed[mask_seed], y_seed[mask_seed])
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * int(mask_seed.sum())
        total_examples += int(mask_seed.sum())
    return total_loss / max(total_examples, 1)
# 학습 루프 정의

@torch.no_grad()
def eval_sage(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        out_seed = out[:batch.batch_size]
        y_seed = batch.y[:batch.batch_size]
        mask_seed = y_seed >= 0
        if mask_seed.sum() == 0:
            continue
        probs = F.softmax(out_seed[mask_seed], dim=1)[:, 1].cpu().numpy()
        labels = y_seed[mask_seed].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    if not all_probs:
        return None, None
    probs_concat = np.concatenate(all_probs)
    labels_concat = np.concatenate(all_labels)
    if len(np.unique(labels_concat)) < 2:
        return None, None
    roc = roc_auc_score(labels_concat, probs_concat)
    pr  = average_precision_score(labels_concat, probs_concat)
    return roc, pr
# ROC-AUC와 PR-AUC 계산

num_epochs = 30
best_val_pr = 0.0
best_state = None
for epoch in range(1, num_epochs + 1):
    loss = train_sage_one_epoch(model, train_loader, optimizer)
    val_roc, val_pr = eval_sage(model, val_loader)
    if val_pr is not None and val_pr > best_val_pr:
        best_val_pr = val_pr
        best_state = model.state_dict()
    if epoch % 5 == 0:
        print(f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val ROC: {val_roc:.4f} | Val PR: {val_pr:.4f}")
# 전체 학습 루프 구성

if best_state is not None:
    model.load_state_dict(best_state)
test_roc, test_pr = eval_sage(model, test_loader)
print(f"GraphSAGE Test ROC-AUC: {test_roc:.4f} | Test PR-AUC: {test_pr:.4f}")
# 최적 모델 로드