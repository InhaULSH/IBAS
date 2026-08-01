# %% Elliptic++ 시간 확장 이종 그래프 기반 GNN 구현
# Elliptic 원본은 트랜잭션 노드 및 트랜잭션 간 자금 흐름 엣지만 있는 단일 그래프, Elliptic++는 여기에 지갑 주소 계층을 추가한 이종 그래프
# : 약 203k 트랜잭션 노드 + 약 822k 지갑 주소 노드 + ddr–addr(주소–주소) 엣지 + addr–tx–addr(주소–트랜잭션–주소) 엣지 + 각 지갑 주소
# 에 대해 56개 피처 + 각 트랜잭션에 대해 183개 피처 + 3개의 라벨이 주소와 트랜잭션 모두에 존재 => 이종 그래프 기반 GNN 적용 시 트랜잭션은
# 실제 자금 이동 사건이고 주소는 그 사건에 등장하는 행위자라는 역할 분리를 명확히 할 수 있음, 결과적으로 주소 레벨 노드 분류를 통해 이 주소가
# illicit actor인지 아닌지 판별하고 서브 그래프 레벨 해석을 통해 특정 주소 - 트랜잭션 - 주소 구조가 이상 거래패턴인지 판단 가능, 따라서 사
# 기 거래 자체뿐만 아니라 사기 거래에서 특정 역할을 하고 있는 행위자 탐지 또한 가능해짐
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
# 라이브러리 즌비
DATA_DIR = "DataSet_DeepLearning/Elliptic++"
wallet_feat_path  = os.path.join(DATA_DIR, "wallets_features.csv")
wallet_class_path = os.path.join(DATA_DIR, "wallets_classes.csv")
addr_addr_path    = os.path.join(DATA_DIR, "AddrAddr_edgelist.csv")
addr_tx_path      = os.path.join(DATA_DIR, "AddrTx_edgelist.csv")
tx_addr_path      = os.path.join(DATA_DIR, "TxAddr_edgelist.csv")
tx_feat_path      = os.path.join(DATA_DIR, "txs_features.csv")
wallet_feat_df  = pd.read_csv(wallet_feat_path)
wallet_class_df = pd.read_csv(wallet_class_path)
addr_addr_df    = pd.read_csv(addr_addr_path)
addr_tx_df      = pd.read_csv(addr_tx_path)
tx_addr_df      = pd.read_csv(tx_addr_path)
tx_feat_df      = pd.read_csv(tx_feat_path)
# 데이터셋 로드

# Elliptic++ 데이터에는 여러 개의 타임 스텝이 존재하므로 한 가지 주소에 대해 여러 행이 존재, 따라서 그대로 이종 그래프 기반 GNN에 사용하기
# 는 어려움 => 주소와 타임스텝 쌍으로 구분되는 address_time 노드를 사용하여 주소 - 트랜젝션 - 주소 그래프를 생성한 다음 address_time 노
# 드를 고유 address 노드와 연결하여 address 노드에 대하여 사기 거래 행위자 탐지 실시, 이때 address 노드는 address_time로부터 정보 집계
wallet_feat_df = wallet_feat_df.sort_values(['address', 'Time step']).reset_index(drop=True)
wallet_feat_df['occurrence'] = wallet_feat_df.groupby(['address', 'Time step']).cumcount() + 1
wallet_feat_df['address_time_idx'] = np.arange(len(wallet_feat_df), dtype=np.int64)
addr_time_feat_cols = [c for c in wallet_feat_df.columns if c not in ['address', 'address_time_idx']]
wallet_feat_df[addr_time_feat_cols] = wallet_feat_df[addr_time_feat_cols].replace([np.inf, -np.inf], np.nan)
addr_time_missing_cols = [c for c in addr_time_feat_cols if wallet_feat_df[c].isna().any()]
if addr_time_missing_cols:
    addr_time_missing_flags = wallet_feat_df[addr_time_missing_cols].isna().astype(np.float32)
    addr_time_missing_flags.columns = [f'{c}_missing' for c in addr_time_missing_flags.columns]
    wallet_feat_df[addr_time_feat_cols] = wallet_feat_df[addr_time_feat_cols].fillna(
        wallet_feat_df[addr_time_feat_cols].median(numeric_only=True)
    ).fillna(0)
    wallet_feat_df = pd.concat([wallet_feat_df, addr_time_missing_flags], axis=1)
    addr_time_feat_cols = [
        c for c in wallet_feat_df.columns if c not in ['address', 'address_time_idx']
    ]
# wallets_features.csv의 각 행을 address_time 노드로 사용, 같은 address와 Time step에 여러 행이 있으면 번호를 붙여 다른 노드로 보존
all_addr_ids_sorted = np.sort(wallet_class_df['address'].unique())
addrid_to_idx = {aid: i for i, aid in enumerate(all_addr_ids_sorted)}
wallet_feat_df['address_idx'] = wallet_feat_df['address'].map(addrid_to_idx)
assert wallet_feat_df['address_idx'].notna().all(), "wallet feature contains addresses missing from class table"
wallet_feat_df['address_idx'] = wallet_feat_df['address_idx'].astype(np.int64)
# 고유 address 노드는 최종 행위자 분류 대상 : 초기 피처는 address_time 피처의 평균값으로 두어 원본 행 피처를 유지하며 안정적으로 초기화
all_txIds_sorted = np.sort(tx_feat_df['txId'].unique())
txid_to_idx = {tid: i for i, tid in enumerate(all_txIds_sorted)}
tx_feat_df = tx_feat_df.set_index('txId').loc[all_txIds_sorted]
tx_feat_cols = list(tx_feat_df.columns)
tx_feat_df[tx_feat_cols] = tx_feat_df[tx_feat_cols].replace([np.inf, -np.inf], np.nan)
tx_missing_cols = [c for c in tx_feat_cols if tx_feat_df[c].isna().any()]
if tx_missing_cols:
    tx_missing_flags = tx_feat_df[tx_missing_cols].isna().astype(np.float32)
    tx_missing_flags.columns = [f'{c}_missing' for c in tx_missing_flags.columns]
    tx_feat_df[tx_feat_cols] = tx_feat_df[tx_feat_cols].fillna(
        tx_feat_df[tx_feat_cols].median(numeric_only=True)
    ).fillna(0)
    tx_feat_df = pd.concat([tx_feat_df, tx_missing_flags], axis=1)
    tx_feat_cols = list(tx_feat_df.columns)
# transaction 노드 :  txs_features.csv의 txId 단위로 만들고, 결측값은 중앙값으로 채운 뒤 결측 여부를 별도 피처로 남김, txId와 Time s
# tep은 엣지 연결 및 시간 분할용 메타데이터로도 따로 보존
wallet_class_df = wallet_class_df.drop_duplicates(subset='address', keep='last').set_index('address')
label_map_addr = {1: 1, 2: 0}
addr_y = torch.full((len(all_addr_ids_sorted),), -1, dtype=torch.long)
for i, aid in enumerate(all_addr_ids_sorted):
    if aid in wallet_class_df.index:
        cls_val = int(wallet_class_df.loc[aid, 'class'])
        if cls_val in label_map_addr:
            addr_y[i] = label_map_addr[cls_val]
# address 라벨 구성
global_address_time_table = wallet_feat_df.groupby('address')['Time step'].agg(
    time_step_first='min',
    time_step_last='max',
    time_step_count='nunique',
).reindex(all_addr_ids_sorted).fillna(0)
global_address_time_step_table = (
    wallet_feat_df[['address', 'Time step']]
    .drop_duplicates()
    .assign(value=1)
    .pivot(index='address', columns='Time step', values='value')
    .fillna(0)
    .astype(bool)
    .reindex(index=all_addr_ids_sorted, columns=range(1, 50), fill_value=False)
)
# 전체 타임스텝 데이터 보존 : 각 address가 전체 타임 스텝 중 언제 등장했는지 테이블 형태로 보존 이후 학습/검증/테스트 데이터셋 분할 시 활용
# 각 노드별 데이터 전처리

def tensor_edge(src, dst):
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        np.vstack([src.astype(np.int64), dst.astype(np.int64)]),
        dtype=torch.long,
    )
# PyG에 맞는 엣지 반환용 헬퍼 함수 : 빈 엣지도 (2, 0) 형태로 안전하게 반환하여 오류 방지

# 시계열 정보가 포함되어있으므로 타입 스텝에 따라 학습/검증/테스트 데이터를 분할해야하나 단순히 구간별로 나눌 경우 검증/테스트 시 학습 시점의
# 정보는 소실됨, 따라서 미래 정보가 현재 단계의 정보와 섞이지 않으면서도 과거의 정보 자체는 활용할 수 있어야함 : 특정 스텝까지 관측 가능한 a
# ddress_time/transaction만 남겨 누적 시간 스냅샷 그래프를 생성, 각 스냅샷 그래프에 대해서 앞 시점 평가에 사용되지 않은 노드와 엣지를 대
# 상으로만 평가를 실시하면 과거 정보에 접근 가능하면서도 미래 정보가 현재 시점 평가에 섞이지 않게할 수 있음
def build_temporal_hetero_graph(max_step):
    # 1~max_step까지만 누적한 이종 그래프를 만듦 : train에는 35~49, validation에는 43~49 정보가 들어가지 않아 시간 정보의 누수를 억제
    snapshot_wallet_df = wallet_feat_df[wallet_feat_df['Time step'] <= max_step].copy()
    snapshot_wallet_df = snapshot_wallet_df.reset_index(drop=True)
    snapshot_wallet_df['snapshot_address_time_idx'] = np.arange(len(snapshot_wallet_df), dtype=np.int64)
    snapshot_tx_df = tx_feat_df[tx_feat_df['Time step'] <= max_step].copy()
    snapshot_tx_ids = snapshot_tx_df.index.values
    snapshot_txid_to_idx = {tid: i for i, tid in enumerate(snapshot_tx_ids)}
    snapshot_tx_time_df = snapshot_tx_df[['Time step']].reset_index()
    snapshot_tx_time_df['transaction_idx'] = snapshot_tx_time_df['txId'].map(snapshot_txid_to_idx).astype(np.int64)
    # 스냅샷 안에서 관측된 address_time 피처만 평균내어 고유 address 노드의 초기 피처로 사용
    # 아직 관측되지 않은 address는 0 피처를 갖지만 라벨 seed로 쓰이지 않으면 손실 계산에는 미포함
    snapshot_address_x_df = (
        snapshot_wallet_df
        .groupby('address')[addr_time_feat_cols]
        .mean()
        .reindex(all_addr_ids_sorted)
        .fillna(0)
    )
    snapshot_address_time_table = snapshot_wallet_df.groupby('address')['Time step'].agg(
        time_step_first='min',
        time_step_last='max',
        time_step_count='nunique',
    ).reindex(all_addr_ids_sorted).fillna(0)
    snapshot_address_time_step_table = (
        snapshot_wallet_df[['address', 'Time step']]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index='address', columns='Time step', values='value')
        .fillna(0)
        .astype(bool)
        .reindex(index=all_addr_ids_sorted, columns=range(1, 50), fill_value=False)
    )
    # HeteroData 노드 구성 : address_time은 원본 wallet 행 단위 노드이며, Time step과 occurrence를 보존해 중복 행도 별도 노드로 유
    # 지, transaction과 address도 같은 스냅샷 기준으로 피처와 시간 메타데이터를 PyG 텐서로 구성
    snapshot_data = HeteroData()
    snapshot_data['address_time'].x = torch.tensor(
        snapshot_wallet_df[addr_time_feat_cols].values,
        dtype=torch.float,
    )
    snapshot_data['address_time'].time_step = torch.tensor(
        snapshot_wallet_df['Time step'].values,
        dtype=torch.long,
    )
    snapshot_data['address_time'].occurrence = torch.tensor(
        snapshot_wallet_df['occurrence'].values,
        dtype=torch.long,
    )
    snapshot_data['address_time'].address_idx = torch.tensor(
        snapshot_wallet_df['address_idx'].values,
        dtype=torch.long,
    )
    snapshot_data['address_time'].time_step_mask = F.one_hot(
        snapshot_data['address_time'].time_step - 1,
        num_classes=49,
    ).to(torch.bool)
    snapshot_data['transaction'].x = torch.tensor(snapshot_tx_df[tx_feat_cols].values, dtype=torch.float)
    snapshot_data['transaction'].time_step = torch.tensor(snapshot_tx_df['Time step'].values, dtype=torch.long)
    snapshot_data['transaction'].time_step_mask = F.one_hot(
        snapshot_data['transaction'].time_step - 1,
        num_classes=49,
    ).to(torch.bool)
    snapshot_data['address'].x = torch.tensor(snapshot_address_x_df.values, dtype=torch.float)
    snapshot_data['address'].time_step_first = torch.tensor(
        snapshot_address_time_table['time_step_first'].values,
        dtype=torch.long,
    )
    snapshot_data['address'].time_step_last = torch.tensor(
        snapshot_address_time_table['time_step_last'].values,
        dtype=torch.long,
    )
    snapshot_data['address'].time_step_count = torch.tensor(
        snapshot_address_time_table['time_step_count'].values,
        dtype=torch.long,
    )
    snapshot_data['address'].time_step_mask = torch.tensor(
        snapshot_address_time_step_table.values,
        dtype=torch.bool,
    )
    snapshot_data['address'].global_time_step_first = torch.tensor(
        global_address_time_table['time_step_first'].values,
        dtype=torch.long,
    )
    snapshot_data['address'].global_time_step_last = torch.tensor(
        global_address_time_table['time_step_last'].values,
        dtype=torch.long,
    )
    snapshot_data['address'].global_time_step_mask = torch.tensor(
        global_address_time_step_table.values,
        dtype=torch.bool,
    )
    snapshot_data['address'].y = addr_y.clone()
    # belongs_to 엣지 : 각 address_time 노드를 고유 address 노드에 연결하여 시점별 행위 정보를 최종 actor 분류 노드로 집계
    belongs_src = snapshot_wallet_df['snapshot_address_time_idx'].values.astype(np.int64)
    belongs_dst = snapshot_wallet_df['address_idx'].values.astype(np.int64)
    snapshot_data['address_time', 'belongs_to', 'address'].edge_index = tensor_edge(belongs_src, belongs_dst)
    # next_time 엣지 : 같은 address의 address_time 노드를 시간 순서대로 연결하여 행동 변화 흐름을 메시지 패싱에 반영
    # 같은 타임스텝의 중복 occurrence도 정렬 순서에 따라 별도 노드로 연결
    next_df = snapshot_wallet_df[['address', 'Time step', 'occurrence', 'snapshot_address_time_idx']].copy()
    next_df = next_df.sort_values(['address', 'Time step', 'occurrence'])
    next_df['next_address_time_idx'] = next_df.groupby('address')['snapshot_address_time_idx'].shift(-1)
    next_df = next_df.dropna(subset=['next_address_time_idx'])
    snapshot_data['address_time', 'next_time', 'address_time'].edge_index = tensor_edge(
        next_df['snapshot_address_time_idx'].astype(np.int64).values,
        next_df['next_address_time_idx'].astype(np.int64).values,
    )
    address_time_lookup_df = snapshot_wallet_df[['address', 'Time step', 'snapshot_address_time_idx']]
    # addr_tx / tx_addr 엣지
    def make_snapshot_addr_tx_edge(edge_df, address_col, tx_col, edge_name):
        # AddrTx/TxAddr는 txId의 Time step을 기준으로 같은 시점의 address_time 노드와 연결
        # max_step 밖의 transaction은 snapshot_tx_time_df에 없으므로 현재 스냅샷 그래프에서 자동 제외
        edge_with_time = edge_df.reset_index(names='edge_row_id').merge(
            snapshot_tx_time_df[['txId', 'Time step', 'transaction_idx']],
            left_on=tx_col,
            right_on='txId',
            how='inner',
        )
        matched = edge_with_time.merge(
            address_time_lookup_df,
            left_on=[address_col, 'Time step'],
            right_on=['address', 'Time step'],
            how='inner',
        )
        dropped = edge_with_time['edge_row_id'].nunique() - matched['edge_row_id'].nunique()
        if dropped > 0:
            print(f"{edge_name} <= {max_step}: dropped {dropped} rows without matching address_time nodes")
        return matched
    addr_tx_matched = make_snapshot_addr_tx_edge(addr_tx_df, 'input_address', 'txId', 'addr_tx')
    snapshot_data['address_time', 'addr_tx', 'transaction'].edge_index = tensor_edge(
        addr_tx_matched['snapshot_address_time_idx'].astype(np.int64).values,
        addr_tx_matched['transaction_idx'].astype(np.int64).values,
    )
    tx_addr_matched = make_snapshot_addr_tx_edge(tx_addr_df, 'output_address', 'txId', 'tx_addr')
    snapshot_data['transaction', 'tx_addr', 'address_time'].edge_index = tensor_edge(
        tx_addr_matched['transaction_idx'].astype(np.int64).values,
        tx_addr_matched['snapshot_address_time_idx'].astype(np.int64).values,
    )
    # address-level addr_addr 엣지 : txId가 없어 정확한 시점을 복원하기 어려운 AddrAddr는 해당 스냅샷에 등장한 address 사이의 addre
    # ss-level 관계로만 보존, 모든 시간 조합에 복제하지 않아 엣지 폭증과 시간 정보 왜곡을 최소화
    active_address_set = set(snapshot_wallet_df['address'].unique())
    addr_addr_snapshot = addr_addr_df[
        addr_addr_df['input_address'].isin(active_address_set)
        & addr_addr_df['output_address'].isin(active_address_set)
    ]
    addr_src = addr_addr_snapshot['input_address'].map(addrid_to_idx)
    addr_dst = addr_addr_snapshot['output_address'].map(addrid_to_idx)
    addr_valid = addr_src.notna() & addr_dst.notna()
    snapshot_data['address', 'addr_addr', 'address'].edge_index = tensor_edge(
        addr_src[addr_valid].astype(np.int64).values,
        addr_dst[addr_valid].astype(np.int64).values,
    )
    return snapshot_data

def validate_temporal_graph(snapshot_data, name):
    edge_sizes = {
        ('address_time', 'belongs_to', 'address'): (
            snapshot_data['address_time'].num_nodes,
            snapshot_data['address'].num_nodes,
        ),
        ('address_time', 'next_time', 'address_time'): (
            snapshot_data['address_time'].num_nodes,
            snapshot_data['address_time'].num_nodes,
        ),
        ('address_time', 'addr_tx', 'transaction'): (
            snapshot_data['address_time'].num_nodes,
            snapshot_data['transaction'].num_nodes,
        ),
        ('transaction', 'tx_addr', 'address_time'): (
            snapshot_data['transaction'].num_nodes,
            snapshot_data['address_time'].num_nodes,
        ),
        ('address', 'addr_addr', 'address'): (
            snapshot_data['address'].num_nodes,
            snapshot_data['address'].num_nodes,
        ),
    }
    for edge_type, (num_src_nodes, num_dst_nodes) in edge_sizes.items():
        edge_index = snapshot_data[edge_type].edge_index
        if edge_index.numel() == 0:
            print(f"{name} {edge_type}: no valid edges")
            continue
        assert int(edge_index[0].max()) < num_src_nodes, f"{name} {edge_type} source index out of range"
        assert int(edge_index[1].max()) < num_dst_nodes, f"{name} {edge_type} destination index out of range"
    assert torch.isfinite(snapshot_data['address_time'].x).all(), f"{name} address_time features contain NaN or inf"
    assert torch.isfinite(snapshot_data['transaction'].x).all(), f"{name} transaction features contain NaN or inf"
    assert torch.isfinite(snapshot_data['address'].x).all(), f"{name} address features contain NaN or inf"
    print(
        f"{name}: address_time={snapshot_data['address_time'].num_nodes}, "
        f"transaction={snapshot_data['transaction'].num_nodes}, address={snapshot_data['address'].num_nodes}"
    )
# 스냅샷 그래프 검증 헬퍼 함수 : 스냅샷별 edge_index 범위와 피처 유한성을 확인하여 NeighborLoader 학습 전에 전처리 오류를 차단

# 시간 누수를 줄이기 위해 34/42/49 타임스텝까지의 누적 스냅샷 그래프를 각각 생성, 세 그래프는 같은 address 인덱스 체계를 공유하므로 동일한
# 모델로 train/valid/test를 수행 가능
train_data = build_temporal_hetero_graph(34)
val_data = build_temporal_hetero_graph(42)
test_data = build_temporal_hetero_graph(49)
validate_temporal_graph(train_data, "train_graph<=34")
validate_temporal_graph(val_data, "val_graph<=42")
validate_temporal_graph(test_data, "test_graph<=49")
# train/valid/test는 각각 34/42/49 스텝까지 누적한 그래프를 사용 seed address는 전체 등장 시점 기준으로 분할 : 따라서 검증 그래프에는
# 43~49 정보가 없고, 학습 그래프에는 35~49 정보가 없음
global_last_step = test_data['address'].global_time_step_last
labeled_address_mask = test_data['address'].y >= 0
train_mask = (global_last_step >= 1) & (global_last_step <= 34) & labeled_address_mask
val_mask = (global_last_step >= 35) & (global_last_step <= 42) & labeled_address_mask
test_mask = (global_last_step >= 43) & (global_last_step <= 49) & labeled_address_mask
train_data['address'].train_mask = train_mask
val_data['address'].val_mask = val_mask
test_data['address'].test_mask = test_mask
train_idx = torch.where(train_mask)[0]
val_idx = torch.where(val_mask)[0]
test_idx = torch.where(test_mask)[0]
print("Train address seeds:", train_idx.size(0))
print("Val address seeds:", val_idx.size(0))
print("Test address seeds:", test_idx.size(0))

# 각 seed address 주변 이웃을 relation별로 제한해 샘플링, 전체 그래프 대신 필요한 주변 서브그래프만 가져와 메모리 사용량 억제
num_neighbors = {
    ('address_time', 'belongs_to', 'address'): [25, 10],
    ('address_time', 'next_time', 'address_time'): [10, 5],
    ('address_time', 'addr_tx', 'transaction'): [15, 10],
    ('transaction', 'tx_addr', 'address_time'): [15, 10],
    ('address', 'addr_addr', 'address'): [15, 10],
}
batch_size = 1024
train_loader = NeighborLoader(
    train_data,
    input_nodes=('address', train_idx),
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = NeighborLoader(
    val_data,
    input_nodes=('address', val_idx),
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=False,
)
test_loader = NeighborLoader(
    test_data,
    input_nodes=('address', test_idx),
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=False,
)
# NeighborLoader 구성

class HeteroTemporalAMLNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        relations = {
            ('address_time', 'belongs_to', 'address'): SAGEConv((-1, -1), hidden_channels),
            ('address_time', 'next_time', 'address_time'): SAGEConv((-1, -1), hidden_channels),
            ('address_time', 'addr_tx', 'transaction'): SAGEConv((-1, -1), hidden_channels),
            ('transaction', 'tx_addr', 'address_time'): SAGEConv((-1, -1), hidden_channels),
            ('address', 'addr_addr', 'address'): SAGEConv((-1, -1), hidden_channels),
        }
        self.conv1 = HeteroConv(relations, aggr='sum')
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in relations
        }, aggr='sum')
        self.lin_addr = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.lin_addr(x_dict['address'])
model = HeteroTemporalAMLNet(hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
train_labels = train_data['address'].y[train_idx]
class_counts = torch.bincount(train_labels, minlength=2).float()
class_weights = class_counts.sum() / (2.0 * class_counts.clamp_min(1.0))
class_weights = class_weights.to(device)
print("Train class counts:", class_counts.tolist())
print("Class weights:", class_weights.detach().cpu().tolist())
# 이종 그래프 기반 GNN 모델 정의 : 관계별 메시지 패싱으로 address_time-transaction-address_time 경로와 시간 흐름을 학습, belongs_to
# 엣지를 통해 시점별 정보를 고유 address 노드로 모아 illicit actor 여부를 예측

# NeighborLoader가 만든 미니 배치에서 앞쪽 batch_size개의 address seed만 손실 계산에 사용, 이웃 address/address_time/transaction
# 노드는 메시지 전달에는 쓰이지만 정답 라벨로 직접 학습하지 않음
def train_temporal_one_epoch(model, loader, optimizer, class_weights):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out_addr = model(batch.x_dict, batch.edge_index_dict)
        seed_count = batch['address'].batch_size
        out_seed = out_addr[:seed_count]
        y_seed = batch['address'].y[:seed_count]
        mask_seed = y_seed >= 0
        if mask_seed.sum() == 0:
            continue
        loss = F.cross_entropy(out_seed[mask_seed], y_seed[mask_seed], weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(mask_seed.sum())
        total_examples += int(mask_seed.sum())
    return total_loss / max(total_examples, 1)
@torch.no_grad()
# 검증/테스트도 seed address만 평가하여, 샘플링된 이웃 노드가 평가 지표에 섞이지 않음
def eval_temporal(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        out_addr = model(batch.x_dict, batch.edge_index_dict)
        seed_count = batch['address'].batch_size
        out_seed = out_addr[:seed_count]
        y_seed = batch['address'].y[:seed_count]
        mask_seed = y_seed >= 0
        if mask_seed.sum() == 0:
            continue
        probs = F.softmax(out_seed[mask_seed], dim=1)[:, 1].detach().cpu().numpy()
        labels = y_seed[mask_seed].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    if not all_probs:
        return None, None
    probs_concat = np.concatenate(all_probs)
    labels_concat = np.concatenate(all_labels)
    if len(np.unique(labels_concat)) < 2:
        return None, None
    roc = roc_auc_score(labels_concat, probs_concat)
    pr = average_precision_score(labels_concat, probs_concat)
    return roc, pr
def metric_text(value):
    return "None" if value is None else f"{value:.4f}"
# 검증 PR-AUC가 가장 높은 모델을 선택한 뒤 후반 타임스텝 test address에 대해 최종 성능을 계산
num_epochs = 30
best_val_pr = -1.0
best_state = None
for epoch in range(1, num_epochs + 1):
    loss = train_temporal_one_epoch(model, train_loader, optimizer, class_weights)
    val_roc, val_pr = eval_temporal(model, val_loader)
    if val_pr is not None and val_pr > best_val_pr:
        best_val_pr = val_pr
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | Loss: {loss:.4f} "
            f"| Val ROC: {metric_text(val_roc)} | Val PR: {metric_text(val_pr)}"
        )
if best_state is not None:
    model.load_state_dict({
        key: value.to(device)
        for key, value in best_state.items()
    })
test_roc, test_pr = eval_temporal(model, test_loader)
print(f"Temporal HeteroGNN Test ROC-AUC: {metric_text(test_roc)} | Test PR-AUC: {metric_text(test_pr)}")