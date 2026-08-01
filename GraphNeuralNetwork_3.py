# GNN 실무 운영
# 모델 자체의 설게뿐만 아니라 이미 학습한 GNN을 금융 실무 시스템 안에서 어떤 역할로 배치할 것인가도 중요 : 모델이 높은 PR-AUC를 냈더라도 그
# 결과가 운영자가 볼 수 있는 위험 점수로 저장되지 않고, 어떤 거래가 왜 위험한지 설명되지 않으며, 대시보드나 알림 시스템으로 이어지지 않는다면
# 실무적으로는 아직 사용 가능한 시스템이 아님 따라서 GNN을 도입한 목적을 고려하여 제한된 비용과 시간을 투입하여 사용자 관점에서 의사결정을 지
# 원하는 시스템을 구축해야함 + 또한 금융 실무에서는 다양한 규칙 기반 탐지 시스템이 작동하고 있으므로 GNN을 최종 결정의 주체로 삼거나 기존 시
# 스템을 모두 대체하기보다는 기존 규칙 기반 탐지 시스템이 탐지하지 못 하는 패턴을 탐지하는 그래프 리스크 엔진 역할을 부여함이 적절(GNN은 주변
# 노드의 정보를 섞어 판단하기 때문에 설명이 어렵고, 데이터 구성 방식에 따라 위험 점수가 민감하게 달라질 수 있으며, Non-Deterministic하므로
# 완전한 정답임을 신뢰하기는 어렵기 때문)

# 이러한 고려를 바탕으로 대시보드 시스템을 다음과 같은 과정으로 설계한다 : 원천 데이터 가공(원천 데이터는 그대로 GNN에 들어갈 수 없으므로 먼
# 저 어떤 것을 노드로 볼지 어떤 것을 엣지로 볼지 결정해야함, 단일 그래프와 이종 그래프간 표현력과 구현/운영 비용 사이의 트레이드 오프도 고려
# 하여 결정해야함) => 위험 점수 계산(모델 출력은 보통 각 노드에 대한 클래스별 logit 또는 확률이나, 이 값을 그대로 쓰지 않고 risk_score로
# 정리하여 위험 등급을 부여하고 어떤 모델 버전에서 나온 점수인지 어떤 데이터 스냅샷을 기준으로 계산했는지 언제 계산했는지 등을 함께 저장) =>
# 해석용 피처 정의(특정 주소의 1-hop 이웃 중 고위험 노드가 몇 개인지 2-hop 안에 불법 라벨 노드가 있는지 최근 타임스텝에서 위험 점수가 상승
# 했는지 연결된 컴포넌트 크기가 급격히 커졌는지 같은 정보를 계산, 대시보드와 운영 판단을 보조하고 해당 주소가 어떤 고위험 주소들과 연결되있는
# 가를 설명) => 대시보드 구현(전체 위험 현황/고위험 거래 목록/고위험 주소 목록/특정 노드의 주변 그래프/위험 사유 요약/시간별 위험도 변화/알
# 림 후보 등 정보를 표시, 모델 학습과 스코어링은 별도 파이프라인에서 수행하고 대시보드는 그 결과 테이블을 읽어 시각화)

# 이때 데이터 구조의 경우 위험 점수 계산 단계의 결과물을 위험 점수 테이블에 저장하여 엔티티별 모델 정보/위험 점수/위험 등급/타임 스텝/최대 스
# 냅샷 등의 정보를 저장하고, 해석용 피처 정의 단계에서 정의한 정보들을 해석용 테이블에 저장하여 노드별 연결성/연결된 엣지수 동향/고위험 연결수
# /연결의 크기 등 정보를 저장하여 운영자의 의사결정을 지원하도록 하는 것이 효율적, 또한 실시간 처리와 배치 처리의 경우 GNN은 주변 그래프가 필
# 요하므로 새 거래 하나가 들어왔을 때 실시간으로 그 거래와 연결된 계좌/주소/디바이스/과거 거래/이웃의 이웃까지 가져오기는 어려우며 지연시간 관
# 리도 까다로움, 따라서 실시간 경로에는 기존 규칙 기반 시스템이나 가벼운 테이블 모델을 두고 GNN은 배치 스코어링 구조를 기본으로 잡아 주기적으
# 로 네트워크 위험도를 갱신하는 구조가 더 효율적, 이렇게 하면 GNN의 강점인 관계 분석을 살리면서도 실시간 서비스 지연 문제를 피할 수 있음

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

try:
    import networkx as nx
except ImportError:
    nx = None
# 라이브러리 준비
@dataclass
class OpsConfig:
    # 경로 설정
    base_dir: Path = Path("DataSet_DeepLearning")
    data_dir_name: str = "Elliptic++"
    output_dir_name: str = "GNN_FraudDashboard_Output"
    # Elliptic++ 파일명
    wallet_feat_file: str = "wallets_features.csv"
    wallet_class_file: str = "wallets_classes.csv"
    addr_addr_file: str = "AddrAddr_edgelist.csv"
    addr_tx_file: str = "AddrTx_edgelist.csv"
    tx_addr_file: str = "TxAddr_edgelist.csv"
    tx_feat_file: str = "txs_features.csv"
    # Elliptic 계열 데이터의 일반적인 라벨 정의: 1=illicit, 2=licit.
    # 사용 중인 데이터셋에서 반대로 정의되어 있다면 여기만 바꾸면 된다.
    illicit_raw_label: int = 1
    licit_raw_label: int = 2
    # 시간 분할 기준
    train_max_step: int = 34
    val_max_step: int = 42
    test_max_step: int = 49
    # 모델 설정
    hidden_channels: int = 64
    out_channels: int = 2
    dropout: float = 0.0
    lr: float = 0.01
    weight_decay: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 1024
    force_retrain: bool = False
    # 대시보드/설명 준비 설정
    model_name: str = "TemporalHeteroGraphSAGE"
    model_version: str = "hetero_sage_ops_v1"
    graph_snapshot_id: str = "ellipticpp_snapshot_49"
    top_k_values: Tuple[int, ...] = (50, 100, 200, 500)
    explanation_candidate_top_n: int = 100
    ego_top_n: int = 20
    max_ego_neighbors_per_seed: int = 50
    # 재현성
    seed: int = 42
    # 경로 추상화
    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.data_dir_name
    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.output_dir_name
    @property
    def model_path(self) -> Path:
        return self.output_dir / "models" / f"{self.model_version}.pt"
    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "run_metadata.json"
CONFIG = OpsConfig()
# relation별 이웃 샘플링 수
NUM_NEIGHBORS = {
    ("address_time", "belongs_to", "address"): [25, 10],
    ("address_time", "next_time", "address_time"): [10, 5],
    ("address_time", "addr_tx", "transaction"): [15, 10],
    ("transaction", "tx_addr", "address_time"): [15, 10],
    ("address", "addr_addr", "address"): [15, 10],
}
# Seed 고정 헬퍼 함수
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# device 설정 헬퍼 함수
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
# 코드 기초 설정

def ensure_dirs(config: OpsConfig) -> None:
    for subdir in [
        config.output_dir,
        config.output_dir / "models",
        config.output_dir / "scores",
        config.output_dir / "reports",
        config.output_dir / "explain",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
# 출력 경로 상에 시스템 운영에 필요한 디렉토리를 설정하는 헬퍼 함수
def save_table(df: pd.DataFrame, path_without_ext: Path) -> None:
    csv_path = path_without_ext.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(path_without_ext.with_suffix(".parquet"), index=False)
    except Exception:
        pass
# 데이터 프레임 저장 헬퍼 함수 : CSV로 항상 저장하고 pyarrow/fastparquet가 있으면 parquet도 추가 저장

class EllipticPPTemporalBuilder:
    def __init__(self, config: OpsConfig):
        self.config = config
        self.wallet_feat_df: Optional[pd.DataFrame] = None
        self.wallet_class_df: Optional[pd.DataFrame] = None
        self.addr_addr_df: Optional[pd.DataFrame] = None
        self.addr_tx_df: Optional[pd.DataFrame] = None
        self.tx_addr_df: Optional[pd.DataFrame] = None
        self.tx_feat_df: Optional[pd.DataFrame] = None

        self.addr_time_feat_cols: List[str] = []
        self.tx_feat_cols: List[str] = []
        self.all_addr_ids_sorted: Optional[np.ndarray] = None
        self.addrid_to_idx: Dict = {}
        self.all_tx_ids_sorted: Optional[np.ndarray] = None
        self.txid_to_idx: Dict = {}
        self.addr_y: Optional[torch.Tensor] = None
        self.global_address_time_table: Optional[pd.DataFrame] = None
        self.global_address_time_step_table: Optional[pd.DataFrame] = None

    def load_raw_csv(self) -> None:
        data_dir = self.config.data_dir
        required = [
            self.config.wallet_feat_file,
            self.config.wallet_class_file,
            self.config.addr_addr_file,
            self.config.addr_tx_file,
            self.config.tx_addr_file,
            self.config.tx_feat_file,
        ]
        missing = [name for name in required if not (data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"다음 Elliptic++ 파일을 찾을 수 없습니다: {missing}\n"
                f"현재 DATA_DIR: {data_dir}\n"
                "CONFIG.data_dir_name 또는 파일명을 확인하십시오."
            )

        self.wallet_feat_df = pd.read_csv(data_dir / self.config.wallet_feat_file)
        self.wallet_class_df = pd.read_csv(data_dir / self.config.wallet_class_file)
        self.addr_addr_df = pd.read_csv(data_dir / self.config.addr_addr_file)
        self.addr_tx_df = pd.read_csv(data_dir / self.config.addr_tx_file)
        self.tx_addr_df = pd.read_csv(data_dir / self.config.tx_addr_file)
        self.tx_feat_df = pd.read_csv(data_dir / self.config.tx_feat_file)

    def prepare_global_tables(self) -> None:
        assert self.wallet_feat_df is not None
        assert self.wallet_class_df is not None
        assert self.tx_feat_df is not None

        wallet_feat_df = self.wallet_feat_df.sort_values(["address", "Time step"]).reset_index(drop=True)
        wallet_feat_df["occurrence"] = wallet_feat_df.groupby(["address", "Time step"]).cumcount() + 1
        wallet_feat_df["address_time_idx"] = np.arange(len(wallet_feat_df), dtype=np.int64)

        self.addr_time_feat_cols = [c for c in wallet_feat_df.columns if c not in ["address", "address_time_idx"]]
        wallet_feat_df[self.addr_time_feat_cols] = wallet_feat_df[self.addr_time_feat_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        missing_cols = [c for c in self.addr_time_feat_cols if wallet_feat_df[c].isna().any()]
        if missing_cols:
            missing_flags = wallet_feat_df[missing_cols].isna().astype(np.float32)
            missing_flags.columns = [f"{c}_missing" for c in missing_flags.columns]
            wallet_feat_df[self.addr_time_feat_cols] = wallet_feat_df[self.addr_time_feat_cols].fillna(
                wallet_feat_df[self.addr_time_feat_cols].median(numeric_only=True)
            ).fillna(0)
            wallet_feat_df = pd.concat([wallet_feat_df, missing_flags], axis=1)
            self.addr_time_feat_cols = [c for c in wallet_feat_df.columns if c not in ["address", "address_time_idx"]]

        self.all_addr_ids_sorted = np.sort(self.wallet_class_df["address"].unique())
        self.addrid_to_idx = {aid: i for i, aid in enumerate(self.all_addr_ids_sorted)}
        wallet_feat_df["address_idx"] = wallet_feat_df["address"].map(self.addrid_to_idx)
        if not wallet_feat_df["address_idx"].notna().all():
            raise ValueError("wallet feature에 class table에 없는 address가 포함되어 있습니다.")
        wallet_feat_df["address_idx"] = wallet_feat_df["address_idx"].astype(np.int64)
        self.wallet_feat_df = wallet_feat_df

        self.all_tx_ids_sorted = np.sort(self.tx_feat_df["txId"].unique())
        self.txid_to_idx = {tid: i for i, tid in enumerate(self.all_tx_ids_sorted)}
        tx_feat_df = self.tx_feat_df.set_index("txId").loc[self.all_tx_ids_sorted]
        self.tx_feat_cols = list(tx_feat_df.columns)
        tx_feat_df[self.tx_feat_cols] = tx_feat_df[self.tx_feat_cols].replace([np.inf, -np.inf], np.nan)
        tx_missing_cols = [c for c in self.tx_feat_cols if tx_feat_df[c].isna().any()]
        if tx_missing_cols:
            tx_missing_flags = tx_feat_df[tx_missing_cols].isna().astype(np.float32)
            tx_missing_flags.columns = [f"{c}_missing" for c in tx_missing_flags.columns]
            tx_feat_df[self.tx_feat_cols] = tx_feat_df[self.tx_feat_cols].fillna(
                tx_feat_df[self.tx_feat_cols].median(numeric_only=True)
            ).fillna(0)
            tx_feat_df = pd.concat([tx_feat_df, tx_missing_flags], axis=1)
            self.tx_feat_cols = list(tx_feat_df.columns)
        self.tx_feat_df = tx_feat_df

        wallet_class = self.wallet_class_df.drop_duplicates(subset="address", keep="last").set_index("address")
        raw_to_binary = {
            self.config.illicit_raw_label: 1,
            self.config.licit_raw_label: 0,
        }
        addr_y = torch.full((len(self.all_addr_ids_sorted),), -1, dtype=torch.long)
        for i, aid in enumerate(self.all_addr_ids_sorted):
            if aid in wallet_class.index:
                raw_label = int(wallet_class.loc[aid, "class"])
                if raw_label in raw_to_binary:
                    addr_y[i] = raw_to_binary[raw_label]
        self.addr_y = addr_y

        self.global_address_time_table = wallet_feat_df.groupby("address")["Time step"].agg(
            time_step_first="min",
            time_step_last="max",
            time_step_count="nunique",
        ).reindex(self.all_addr_ids_sorted).fillna(0)

        self.global_address_time_step_table = (
            wallet_feat_df[["address", "Time step"]]
            .drop_duplicates()
            .assign(value=1)
            .pivot(index="address", columns="Time step", values="value")
            .fillna(0)
            .astype(bool)
            .reindex(index=self.all_addr_ids_sorted, columns=range(1, 50), fill_value=False)
        )

    @staticmethod
    def tensor_edge(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
        if len(src) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(np.vstack([src.astype(np.int64), dst.astype(np.int64)]), dtype=torch.long)

    def build_temporal_hetero_graph(self, max_step: int) -> HeteroData:
        assert self.wallet_feat_df is not None
        assert self.tx_feat_df is not None
        assert self.addr_addr_df is not None
        assert self.addr_tx_df is not None
        assert self.tx_addr_df is not None
        assert self.all_addr_ids_sorted is not None
        assert self.addr_y is not None
        assert self.global_address_time_table is not None
        assert self.global_address_time_step_table is not None

        snapshot_wallet_df = self.wallet_feat_df[self.wallet_feat_df["Time step"] <= max_step].copy().reset_index(drop=True)
        snapshot_wallet_df["snapshot_address_time_idx"] = np.arange(len(snapshot_wallet_df), dtype=np.int64)

        snapshot_tx_df = self.tx_feat_df[self.tx_feat_df["Time step"] <= max_step].copy()
        snapshot_tx_ids = snapshot_tx_df.index.values
        snapshot_txid_to_idx = {tid: i for i, tid in enumerate(snapshot_tx_ids)}
        snapshot_tx_time_df = snapshot_tx_df[["Time step"]].reset_index()
        snapshot_tx_time_df["transaction_idx"] = snapshot_tx_time_df["txId"].map(snapshot_txid_to_idx).astype(np.int64)

        snapshot_address_x_df = (
            snapshot_wallet_df.groupby("address")[self.addr_time_feat_cols]
            .mean()
            .reindex(self.all_addr_ids_sorted)
            .fillna(0)
        )
        snapshot_address_time_table = snapshot_wallet_df.groupby("address")["Time step"].agg(
            time_step_first="min",
            time_step_last="max",
            time_step_count="nunique",
        ).reindex(self.all_addr_ids_sorted).fillna(0)
        snapshot_address_time_step_table = (
            snapshot_wallet_df[["address", "Time step"]]
            .drop_duplicates()
            .assign(value=1)
            .pivot(index="address", columns="Time step", values="value")
            .fillna(0)
            .astype(bool)
            .reindex(index=self.all_addr_ids_sorted, columns=range(1, 50), fill_value=False)
        )

        data = HeteroData()
        data["address_time"].x = torch.tensor(snapshot_wallet_df[self.addr_time_feat_cols].values, dtype=torch.float)
        data["address_time"].time_step = torch.tensor(snapshot_wallet_df["Time step"].values, dtype=torch.long)
        data["address_time"].occurrence = torch.tensor(snapshot_wallet_df["occurrence"].values, dtype=torch.long)
        data["address_time"].address_idx = torch.tensor(snapshot_wallet_df["address_idx"].values, dtype=torch.long)
        data["address_time"].time_step_mask = F.one_hot(data["address_time"].time_step - 1, num_classes=49).to(torch.bool)

        data["transaction"].x = torch.tensor(snapshot_tx_df[self.tx_feat_cols].values, dtype=torch.float)
        data["transaction"].time_step = torch.tensor(snapshot_tx_df["Time step"].values, dtype=torch.long)
        data["transaction"].original_txid = torch.tensor(snapshot_tx_ids.astype(np.int64), dtype=torch.long)
        data["transaction"].time_step_mask = F.one_hot(data["transaction"].time_step - 1, num_classes=49).to(torch.bool)

        data["address"].x = torch.tensor(snapshot_address_x_df.values, dtype=torch.float)
        data["address"].time_step_first = torch.tensor(snapshot_address_time_table["time_step_first"].values, dtype=torch.long)
        data["address"].time_step_last = torch.tensor(snapshot_address_time_table["time_step_last"].values, dtype=torch.long)
        data["address"].time_step_count = torch.tensor(snapshot_address_time_table["time_step_count"].values, dtype=torch.long)
        data["address"].time_step_mask = torch.tensor(snapshot_address_time_step_table.values, dtype=torch.bool)
        data["address"].global_time_step_first = torch.tensor(
            self.global_address_time_table["time_step_first"].values, dtype=torch.long
        )
        data["address"].global_time_step_last = torch.tensor(
            self.global_address_time_table["time_step_last"].values, dtype=torch.long
        )
        data["address"].global_time_step_mask = torch.tensor(
            self.global_address_time_step_table.values, dtype=torch.bool
        )
        data["address"].y = self.addr_y.clone()

        belongs_src = snapshot_wallet_df["snapshot_address_time_idx"].values.astype(np.int64)
        belongs_dst = snapshot_wallet_df["address_idx"].values.astype(np.int64)
        data["address_time", "belongs_to", "address"].edge_index = self.tensor_edge(belongs_src, belongs_dst)

        next_df = snapshot_wallet_df[["address", "Time step", "occurrence", "snapshot_address_time_idx"]].copy()
        next_df = next_df.sort_values(["address", "Time step", "occurrence"])
        next_df["next_address_time_idx"] = next_df.groupby("address")["snapshot_address_time_idx"].shift(-1)
        next_df = next_df.dropna(subset=["next_address_time_idx"])
        data["address_time", "next_time", "address_time"].edge_index = self.tensor_edge(
            next_df["snapshot_address_time_idx"].astype(np.int64).values,
            next_df["next_address_time_idx"].astype(np.int64).values,
        )

        address_time_lookup = snapshot_wallet_df[["address", "Time step", "snapshot_address_time_idx"]]

        def make_snapshot_addr_tx_edge(edge_df: pd.DataFrame, address_col: str, tx_col: str) -> pd.DataFrame:
            edge_with_time = edge_df.reset_index(names="edge_row_id").merge(
                snapshot_tx_time_df[["txId", "Time step", "transaction_idx"]],
                left_on=tx_col,
                right_on="txId",
                how="inner",
            )
            matched = edge_with_time.merge(
                address_time_lookup,
                left_on=[address_col, "Time step"],
                right_on=["address", "Time step"],
                how="inner",
            )
            return matched

        addr_tx_matched = make_snapshot_addr_tx_edge(self.addr_tx_df, "input_address", "txId")
        data["address_time", "addr_tx", "transaction"].edge_index = self.tensor_edge(
            addr_tx_matched["snapshot_address_time_idx"].astype(np.int64).values,
            addr_tx_matched["transaction_idx"].astype(np.int64).values,
        )

        tx_addr_matched = make_snapshot_addr_tx_edge(self.tx_addr_df, "output_address", "txId")
        data["transaction", "tx_addr", "address_time"].edge_index = self.tensor_edge(
            tx_addr_matched["transaction_idx"].astype(np.int64).values,
            tx_addr_matched["snapshot_address_time_idx"].astype(np.int64).values,
        )

        active_address_set = set(snapshot_wallet_df["address"].unique())
        addr_addr_snapshot = self.addr_addr_df[
            self.addr_addr_df["input_address"].isin(active_address_set)
            & self.addr_addr_df["output_address"].isin(active_address_set)
        ]
        addr_src = addr_addr_snapshot["input_address"].map(self.addrid_to_idx)
        addr_dst = addr_addr_snapshot["output_address"].map(self.addrid_to_idx)
        addr_valid = addr_src.notna() & addr_dst.notna()
        data["address", "addr_addr", "address"].edge_index = self.tensor_edge(
            addr_src[addr_valid].astype(np.int64).values,
            addr_dst[addr_valid].astype(np.int64).values,
        )
        return data

    def validate_graph(self, data: HeteroData, name: str) -> None:
        edge_sizes = {
            ("address_time", "belongs_to", "address"): (data["address_time"].num_nodes, data["address"].num_nodes),
            ("address_time", "next_time", "address_time"): (data["address_time"].num_nodes, data["address_time"].num_nodes),
            ("address_time", "addr_tx", "transaction"): (data["address_time"].num_nodes, data["transaction"].num_nodes),
            ("transaction", "tx_addr", "address_time"): (data["transaction"].num_nodes, data["address_time"].num_nodes),
            ("address", "addr_addr", "address"): (data["address"].num_nodes, data["address"].num_nodes),
        }
        for edge_type, (src_n, dst_n) in edge_sizes.items():
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                print(f"{name} {edge_type}: no valid edges")
                continue
            assert int(edge_index[0].max()) < src_n, f"{name} {edge_type} source index out of range"
            assert int(edge_index[1].max()) < dst_n, f"{name} {edge_type} destination index out of range"
        for node_type in ["address_time", "transaction", "address"]:
            assert torch.isfinite(data[node_type].x).all(), f"{name} {node_type} feature contains NaN or Inf"
        print(
            f"{name}: address_time={data['address_time'].num_nodes}, "
            f"transaction={data['transaction'].num_nodes}, address={data['address'].num_nodes}"
        )
# Elliptic++ 전처리와 시간 스냅샷 이종 그래프 구성 : GraphNeuralNetwork_2에서의 동작과 동일
class HeteroTemporalAMLNet(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        relations = {
            ("address_time", "belongs_to", "address"): SAGEConv((-1, -1), hidden_channels),
            ("address_time", "next_time", "address_time"): SAGEConv((-1, -1), hidden_channels),
            ("address_time", "addr_tx", "transaction"): SAGEConv((-1, -1), hidden_channels),
            ("transaction", "tx_addr", "address_time"): SAGEConv((-1, -1), hidden_channels),
            ("address", "addr_addr", "address"): SAGEConv((-1, -1), hidden_channels),
        }
        self.conv1 = HeteroConv(relations, aggr="sum")
        self.conv2 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in relations}, aggr="sum")
        self.lin_addr = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.lin_addr(x_dict["address"])
def attach_temporal_masks(train_data: HeteroData, val_data: HeteroData, test_data: HeteroData, config: OpsConfig):
    global_last_step = test_data["address"].global_time_step_last
    labeled_mask = test_data["address"].y >= 0

    train_mask = (global_last_step >= 1) & (global_last_step <= config.train_max_step) & labeled_mask
    val_mask = (global_last_step >= config.train_max_step + 1) & (global_last_step <= config.val_max_step) & labeled_mask
    test_mask = (global_last_step >= config.val_max_step + 1) & (global_last_step <= config.test_max_step) & labeled_mask

    train_data["address"].train_mask = train_mask
    val_data["address"].val_mask = val_mask
    test_data["address"].test_mask = test_mask

    return torch.where(train_mask)[0], torch.where(val_mask)[0], torch.where(test_mask)[0]
def make_loaders(train_data, val_data, test_data, train_idx, val_idx, test_idx, config: OpsConfig):
    train_loader = NeighborLoader(
        train_data,
        input_nodes=("address", train_idx),
        num_neighbors=NUM_NEIGHBORS,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        val_data,
        input_nodes=("address", val_idx),
        num_neighbors=NUM_NEIGHBORS,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        test_data,
        input_nodes=("address", test_idx),
        num_neighbors=NUM_NEIGHBORS,
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
def compute_class_weights(data: HeteroData, train_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    train_labels = data["address"].y[train_idx]
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts.clamp_min(1.0))
    print("Train class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())
    return class_weights.to(device)
def train_one_epoch(model, loader, optimizer, class_weights, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out_addr = model(batch.x_dict, batch.edge_index_dict)
        seed_count = batch["address"].batch_size
        out_seed = out_addr[:seed_count]
        y_seed = batch["address"].y[:seed_count]
        mask_seed = y_seed >= 0
        if int(mask_seed.sum()) == 0:
            continue
        loss = F.cross_entropy(out_seed[mask_seed], y_seed[mask_seed], weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(mask_seed.sum())
        total_examples += int(mask_seed.sum())
    return total_loss / max(total_examples, 1)
@torch.no_grad()
def eval_loader(model, loader, device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        out_addr = model(batch.x_dict, batch.edge_index_dict)
        seed_count = batch["address"].batch_size
        out_seed = out_addr[:seed_count]
        y_seed = batch["address"].y[:seed_count]
        mask_seed = y_seed >= 0
        if int(mask_seed.sum()) == 0:
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
    return roc_auc_score(labels_concat, probs_concat), average_precision_score(labels_concat, probs_concat)
def initialize_lazy_model(model, loader, device: torch.device) -> None:
    model.eval()
    try:
        batch = next(iter(loader)).to(device)
    except StopIteration:
        raise RuntimeError("NeighborLoader가 비어 있어 모델 lazy layer를 초기화할 수 없습니다.")
    with torch.no_grad():
        _ = model(batch.x_dict, batch.edge_index_dict)
def train_or_load_model(model, train_loader, val_loader, train_data, train_idx, config: OpsConfig, device: torch.device):
    initialize_lazy_model(model, train_loader, device)

    if config.model_path.exists() and not config.force_retrain:
        print(f"Saved model found. Loading: {config.model_path}")
        model.load_state_dict(torch.load(config.model_path, map_location=device))
        return model

    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    class_weights = compute_class_weights(train_data, train_idx, device)
    best_val_pr = -1.0
    best_state = None
    history = []

    for epoch in range(1, config.num_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, class_weights, device)
        val_roc, val_pr = eval_loader(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss, "val_roc_auc": val_roc, "val_pr_auc": val_pr})
        if val_pr is not None and val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val ROC {val_roc} | Val PR {val_pr}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), config.model_path)
    save_table(pd.DataFrame(history), config.output_dir / "reports" / "training_history")
    return model
@torch.no_grad()
def score_loader_to_frame(
    model,
    loader,
    split_name: str,
    snapshot_step: int,
    address_ids_sorted: np.ndarray,
    config: OpsConfig,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    scoring_time = datetime.now(timezone.utc).isoformat()

    for batch in loader:
        batch = batch.to(device)
        out_addr = model(batch.x_dict, batch.edge_index_dict)
        seed_count = batch["address"].batch_size
        logits = out_addr[:seed_count]
        y = batch["address"].y[:seed_count]
        probs = F.softmax(logits, dim=1)[:, 1]

        if hasattr(batch["address"], "n_id"):
            global_idx = batch["address"].n_id[:seed_count].detach().cpu().numpy()
        else:
            # PyG NeighborLoader는 일반적으로 n_id를 제공한다.
            # 만약 제공되지 않는 환경이면 이 분기에서 entity_id를 순번으로 대체한다.
            global_idx = np.arange(seed_count)

        time_last = batch["address"].global_time_step_last[:seed_count].detach().cpu().numpy()
        time_first = batch["address"].global_time_step_first[:seed_count].detach().cpu().numpy()
        time_count = batch["address"].time_step_count[:seed_count].detach().cpu().numpy()

        for i in range(seed_count):
            addr_idx = int(global_idx[i])
            rows.append({
                "entity_type": "address",
                "entity_index": addr_idx,
                "entity_id": str(address_ids_sorted[addr_idx]),
                "split": split_name,
                "snapshot_step": snapshot_step,
                "time_step_first": int(time_first[i]),
                "time_step_last": int(time_last[i]),
                "time_step_count": int(time_count[i]),
                "y_true": int(y[i].detach().cpu()),
                "risk_score": float(probs[i].detach().cpu()),
                "model_name": config.model_name,
                "model_version": config.model_version,
                "graph_snapshot_id": f"ellipticpp_snapshot_{snapshot_step}",
                "scoring_time": scoring_time,
            })

    return pd.DataFrame(rows)
# 모델 학습 및 로드 : GraphNeuralNetwork_2에서의 동작과 유사하나, 첫 실행 시에만 모델을 학습하고 이후에는 GNN_FraudDashboard_Output/
# models에 저장된 모델 파일이 있으면 재학습하지 않고 불러옴, 또한 모델 예측 결과를 데이터프레임으로 저장

def assign_risk_band(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"
# 위험 점수에 따라 위험 구간을 배정하는 함수 : 위험 점수의 스케일과 위험 구간의 스레숄드는 운영자의 해석에 따라 자유롭게 조절 가능
def build_fraud_scores(pred_df: pd.DataFrame) -> pd.DataFrame:
    score_df = pred_df.copy()
    score_df["risk_band"] = score_df["risk_score"].apply(assign_risk_band)
    score_df["is_labeled"] = score_df["y_true"] >= 0
    score_df["is_high_priority"] = score_df["risk_band"].isin(["CRITICAL", "HIGH"])
    return score_df.sort_values(["risk_score", "time_step_last"], ascending=[False, False]).reset_index(drop=True)
# 모델 예측 결과를 대시보드용 위험 점수 테이블로 변환하는 함수 : 엔티티 정보/타입 스텝 정보/모델 정보/스냅샷 정보/위험 점수/점수 부여 시점
# 등 정보를 데이터 프레임 형태로 저장


# %% 시간 누수/라벨 지연/불균형 문제
# 이 세 문제는 금융 사기 거래 탐지에서 불가피한 문제, 거래 데이터는 시간 순서대로 발생하고 사기 여부는 거래 직후 바로 확정되지 않으며 실제 사
# 기 거래는 정상 거래에 비해 극도로 적음, 이를 고려하여 기존 코드 실습을 진행했으나 금융 실무 관점에서 한 번 더 자세히 살펴볼 필요가 있음

# 금융 데이터에서 시간은 단순한 컬럼이 아니다 : 금융 거래는 본질적으로 시간의 흐름 속에서 발생하고 모델이 실제 운영에서 해야할 일은 이미 과거
# 에 발생한 패턴을 바탕으로 아직 모르는 미래의 위험을 예측하는 것, 따라서 학습/검증/테스트 데이터셋의 무작위 분할은 매우 위험, 예를들어 무작
# 위 분할 시 실무에서는 사용할 수 없는 미래 시점의 정보를 과거 시점에 활용하는 것이 가능해나 실제 운영 상황에서는 그러한 행위는 불가능, 따라
# 서 금융 데이터에서 시간은 단순히 피쳐 하나가 아니며 어떤 노드와 엣지가 그 시점에 관측 가능했는지를 결정하는 기준, 어떤 시점의 예측을 수행할
# 때는 그 시점 이전에 존재했던 노드/엣지/피처/라벨만 사용할 수 있어야함

# 보통 이러한 시간 누수는 다음의 3가지 경로로 발생 가능 = 피쳐 누수(예측 시점 이후에만 알 수 있는 정보가 피처에 포함되는 경우, 향후 30일 동
# 안의 총 거래 횟수/이후 신고 접수 여부/사기 확정까지 걸린 시간 등 정보는 학습 피처에 들어가서는 안 됨, 분석용 테이블을 만들때 사후 집계값을
# 붙이다 보면 모델이 운영 시점에는 알 수 없는 정보를 학습할 가능성 발생, 따라서 모델의 학습과 사후 분석 과정은 분리되어야 함) / 라벨 누수(사
# 기 확정 여부나 조사 결과가 직접 또는 간접적으로 피처에 반영되는 경우, 어떤 거래가 사기 조사 대상으로 지정되어 내부 상태 코드가 바뀌었고, 그
# 상태 코드가 모델 피처에 들어갔다면, 모델은 실제 사기 패턴이 아니라 사후 처리 흔적을 학습할 수 있음) / 그래프 구조 누수(GNN에서 특히 중요하
# 며, 어떤 노드의 피처는 과거 기준으로 잘 정리했더라도 엣지 리스트가 전체 기간 기준으로 만들어져 있으면 미래 연결이 과거 예측에 섞일 수 있음,
# 10번째 타임스텝의 거래를 예측하면서 40번째 타임스텝에 생긴 거래와의 연결이 그래프에 포함되어 있다면 메시지 패싱 과정에서 미래 정보가 들어감,
# 따라서 데이터셋 분할 시 마스크만 시간 기준으로 나누기 보다는 시점별 그래프 스냅샷을 따로 생성하여 활용하는 방식이 더 권장됨, 스냅샷 그래프는
# 특정 시점까지 관측된 데이터만 모아 만든 그래프이므로 Elliptic++ 데이터에서의 데이터셋 분할의 경우와 같이, 검증과 테스트에서 과거 정보는 사
# 용할 수 있지만 미래 정보는 사용할 수 없도록 할 수 있기 때문)

# 라벨 지연 : 일반적인 지도학습에서는 각 데이터에 정답 라벨이 있다고 가정하나 금융에서는 거래가 발생한 순간에 사기인지 정상인지 완벽하게 알 수
# 없으며 반대로 정상처럼 보였던 거래가 나중에 사기로 밝혀질 수도 있음, 이 문제는 모델 학습과 평가를 모두 어렵게 만듦, 따라서 금융 모델 운영에
# 서는 라벨이 없는 데이터 또한 자연스럽게 여기고 모델 학습 시 반영하는 접근이 중요하며, 사기 여부가 확정된 시점 전후의 라벨을 나누어 관리하는
# 것 또한 필요할 수 있음, 마지막으로 지연 평가를 통한 평가방식 수립이 필요, 예를 들어 매일 모델이 거래 위험 점수를 계산한다고 할때 5월 1일에
# 계산한 점수는 5월 1일에는 정답이 충분하지 않아 완전한 PR-AUC를 계산하기 어려우므로 5월 15일이나 6월 1일이 되었을 때 5월 1일 거래 중 라벨
# 이 확정된 건들을 모아 뒤늦게 평가하는 방식, 즉 평가 시점에서 라벨이 확정된 과거 예측에 대한 지연 평가 결과만을 평가지표로 사용하는 것

# 클래스 불균형 : 금융 사기 탐지에서 사기 거래는 정상 거래보다 훨씬 적음, 만약 거래의 0.1%만 사기라면 모든 거래를 정상이라고 예측해도 정확도
# 는 99.9%이므로 정확도는 평가 지표로서 의미가 없음, 따라서 ROC-AUC와 PR-AUC를 평가지표로 사용함이 적절, 이때 ROC-AUC는 양성 클래스와 음
# 성 클래스의 순위 구분 능력을 보는 데 유용하지만 극단적 불균형 상황에서는 실제 운영자가 체감하는 성능을 충분히 보여주지 못할 수 있음, 반면 P
# R-AUC는 precision과 recall의 관계를 보기 때문에 모델이 고위험이라고 잡은 것 중 실제 사기가 얼마나 되는가와 전체 사기 중 얼마나 잡았는가
# 를 더 직접적으로 보여줌, 금융 실무에서는 precision과 recall 사이의 균형이 비즈니스 의사결정과 연결되고 precision과 recall 사이의 트레
# 이드 오프를 비용/편익/규제 여뷰에 유연하게 조절하여야할 필요가 있으므로 대개 PR-AUC가 더 유용함

# 모델 측면에서는 ROC-AUC, PR-AUC, F1-score를 중요한 평가지표로 사용하나, 운영 측면에서는 오늘 몇 건이 심사 대상으로 올라오는가?/그중 실
# 제 사기는 몇 건일 가능성이 높은가?/심사 인력 1명이 하루 50건을 볼 수 있다면 몇명이 필요한가?/상위 100건만 보면 전체 사기의 몇 퍼센트를 잡
# 을 수 있는가? 같은 질문이 더 중요한 평가지표임, 따라서 대시보드 시스템 구축과 운영을 위해서는 모델 평가 지표를 운영 지표로 바꾸어야함 = 상
# 위 K건 기준 precision/상위 K건 기준 recall/위험 등급별 건수/위험 등급별 실제 사기 비율/하루 예상 심사 건수/심사 가능 건수 대비 초과량/
# 고위험 알림 중 중복 네트워크 비율 등

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(-scores)[: min(k, len(scores))]
    return float(np.mean(y_true[order])) if len(order) > 0 else 0.0
# precision 계산용 헬퍼함수 : 위험 점수가 높은 상위 K개 주소 중 실제 illicit 주소의 비율
def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    total_positive = float(np.sum(y_true))
    if total_positive <= 0:
        return 0.0
    order = np.argsort(-scores)[: min(k, len(scores))]
    return float(np.sum(y_true[order]) / total_positive)
# recall 계산용 헬퍼함수 : 전체 illicit 주소 중에서 위험 점수 상위 K개 안에 포함된 비율
def evaluate_split(score_df: pd.DataFrame, split_name: str, top_k_values: Iterable[int]) -> Dict:
    part = score_df[(score_df["split"] == split_name) & (score_df["y_true"] >= 0)].copy()
    result = {"split": split_name, "num_labeled": len(part)}
    if len(part) == 0 or part["y_true"].nunique() < 2:
        result.update({"roc_auc": None, "pr_auc": None, "precision": None, "recall": None, "f1": None})
        for k in top_k_values:
            result[f"precision_at_{k}"] = None
            result[f"recall_at_{k}"] = None
        return result
    y_true = part["y_true"].astype(int).values
    scores = part["risk_score"].values
    y_pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    result.update({
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "positive_rate": float(np.mean(y_true)),
        "avg_risk_score": float(np.mean(scores)),
    })
    for k in top_k_values:
        result[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
        result[f"recall_at_{k}"] = recall_at_k(y_true, scores, k)
    return result
# Train/Validation/Test 중 하나의 스플릿에 대해 각종 평가 지표/운영 지표를 계산하는 헬퍼 함수
def build_eval_reports(score_df: pd.DataFrame, config: OpsConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_summary = pd.DataFrame([
        evaluate_split(score_df, split, config.top_k_values)
        for split in ["train", "validation", "test"]
    ])
    labeled = score_df[score_df["y_true"] >= 0].copy()
    band_summary = (
        labeled.groupby(["split", "risk_band"], observed=False)
        .agg(
            count=("entity_id", "count"),
            avg_score=("risk_score", "mean"),
            fraud_rate=("y_true", "mean"),
            max_score=("risk_score", "max"),
            min_score=("risk_score", "min"),
        )
        .reset_index()
    )
    time_summary = (
        labeled.groupby(["split", "time_step_last"], observed=False)
        .agg(
            count=("entity_id", "count"),
            fraud_count=("y_true", "sum"),
            fraud_rate=("y_true", "mean"),
            avg_score=("risk_score", "mean"),
            high_priority_count=("is_high_priority", "sum"),
        )
        .reset_index()
    )
    return eval_summary, band_summary, time_summary
# 스플릿별/위험 등급별/시간 흐름별 각종 평가 지표/운영 지표를 계산하여 모델 성능을 요약하는 함수


# %% 운영 데이터 계층 구축
# GNN 학습이 끝났다고 해서 곧바로 대시보드를 만들 수 있는 것은 아님. 모델이 직접 출력하는 값은 각 address에 대한 클래스별 logit 또는 ill
# icit 확률에 가까우며 이 값만으로는 운영자가 어떤 주소를 우선 확인해야 하는지, 어떤 시점의 그래프와 어떤 모델 버전에서 계산된 결과인지, 해
# 당 주소가 주변 네트워크에서 어떤 관계를 갖는지 알기 어려움

# 따라서 모델 학습 계층과 대시보드 계층 사이에 별도의 '운영 데이터 계층' 을 두고 모델 예측 결과를 위험 점수 테이블과 운영 평가 지표로 표준화
# 하여 저장할 필요가 있음. 이 구조를 사용하면 대시보드는 학습된 모델이나 객체를 직접 불러오지 않고도 정해진 형식의 테이블만 읽어 화면을 구성할
# 수 있음. 이후 모델 구조가 GraphSAGE에서 다른 GNN으로 변경되더라도 위험 점수 테이블의 스키마를 유지하면 대시보드 코드는 크게 수정하지 않아
# 도 되며, 모델 학습 실패나 재학습 중에도 마지막으로 정상 생성된 산출물을 사용해 대시보드를 운영할 수 있음

# 먼저 모델 예측 결과는 address별 risk_score, risk_band, split, snapshot_step, model_version, scoring_time 등을 포함한 위험 점
# 수 테이블로 변환하고, 이를 통해 확률값을 운영자가 이해할 수 있는 위험 등급과 우선순위 정보로 바꾸고, 동일 주소의점수가 언제·어떤 데이터·어떤
# 모델에서 생성되었는지 추적할 수 있게 함, 또한 설명 가능성과 네트워크 시각화에 필요한 다음과 같은 최소 기반 테이블을 생성이 필요함

# address_index_map : PyTorch Geometric 내부에서 사용하는 entity_index와 Elliptic++ 원본 address ID를 연결하는 매핑 테이블, GNN
# 은 계산 효율을 위해 원본 주소 문자열이나 큰 정수 ID를 그대로 사용하지 않고 0부터 시작하는 내부 인덱스로 변환하므로, 모델 결과의 entity_in
# dex만으로는 운영자가 실제 주소를 식별할 수 없음, 따라서 대시보드에서 특정 주소에 대한 접근과 검색이 가능하려면 이 테이블이 반드시 필요
# address_addr_edges : 최종 테스트 스냅샷에서 관측된 address-address 관계를 저장한 엣지 테이블, 위험 점수만으로는 이 주소가 왜 우선 검
# 토 대상인지 충분히 설명하기 어렵기 때문에 이후 장에서는 이 테이블을 이용해 특정 주소의 1-hop 이웃, 주변 고위험 주소 수, illicit 라벨 이웃
# 수, 연결 방향, ego network 등을 계산 가능

def build_address_index_map(address_ids_sorted: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "entity_type": "address",
        "entity_index": np.arange(len(address_ids_sorted), dtype=np.int64),
        "entity_id": [str(x) for x in address_ids_sorted],
    })
def export_address_edge_table(test_data: HeteroData, address_ids_sorted: np.ndarray,
                              config: OpsConfig,) -> pd.DataFrame:
    edge_index = test_data["address", "addr_addr", "address"].edge_index.detach().cpu()
    if edge_index.numel() == 0:
        return pd.DataFrame(columns=[
            "src_index", "dst_index", "src_id", "dst_id", "edge_type", "snapshot_step"
        ])

    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    edge_df = pd.DataFrame({
        "src_index": src,
        "dst_index": dst,
        "src_id": [str(address_ids_sorted[i]) for i in src],
        "dst_id": [str(address_ids_sorted[i]) for i in dst],
        "edge_type": "addr_addr",
        "snapshot_step": config.test_max_step,
    })
    return edge_df.drop_duplicates().reset_index(drop=True)
def build_explain_base_summary(fraud_scores: pd.DataFrame, address_index_map: pd.DataFrame,
                               address_edge_table: pd.DataFrame,) -> pd.DataFrame:
    latest_scores = fraud_scores.sort_values("snapshot_step").drop_duplicates("entity_index", keep="last")
    high_priority = latest_scores[latest_scores["risk_band"].isin(["CRITICAL", "HIGH"])]

    rows = [
        {
            "item": "num_address_index_map_rows",
            "value": int(len(address_index_map)),
            "description": "PyG 내부 address index와 원본 address id를 연결하는 매핑 행 수",
        },
        {
            "item": "num_scored_addresses",
            "value": int(latest_scores["entity_index"].nunique()),
            "description": "현재 fraud_scores에 포함된 고유 address 수",
        },
        {
            "item": "num_high_priority_addresses",
            "value": int(len(high_priority)),
            "description": "CRITICAL 또는 HIGH 등급으로 분류된 address 수",
        },
        {
            "item": "num_address_addr_edges",
            "value": int(len(address_edge_table)),
            "description": "최종 스냅샷에서 관측된 address-address 엣지 수",
        },
        {
            "item": "num_edge_source_addresses",
            "value": int(address_edge_table["src_index"].nunique()) if not address_edge_table.empty else 0,
            "description": "address-address 엣지의 출발점으로 등장한 고유 address 수",
        },
        {
            "item": "num_edge_destination_addresses",
            "value": int(address_edge_table["dst_index"].nunique()) if not address_edge_table.empty else 0,
            "description": "address-address 엣지의 도착점으로 등장한 고유 address 수",
        },
    ]
    return pd.DataFrame(rows)
# explain_base_summary는 대시보드 구축 전 점검을 위한 임시 요약 테이블로 이후 과정에서는 사용되지 않는다
config = CONFIG
set_seed(config.seed)
ensure_dirs(config)
device = get_device()
print("Device:", device)
print("Output directory:", config.output_dir)
builder = EllipticPPTemporalBuilder(config)
builder.load_raw_csv()
builder.prepare_global_tables()
train_data = builder.build_temporal_hetero_graph(config.train_max_step)
val_data = builder.build_temporal_hetero_graph(config.val_max_step)
test_data = builder.build_temporal_hetero_graph(config.test_max_step)
builder.validate_graph(train_data, f"train_graph<={config.train_max_step}")
builder.validate_graph(val_data, f"val_graph<={config.val_max_step}")
builder.validate_graph(test_data, f"test_graph<={config.test_max_step}")
train_idx, val_idx, test_idx = attach_temporal_masks(train_data, val_data, test_data, config)
print("Train address seeds:", train_idx.size(0))
print("Validation address seeds:", val_idx.size(0))
print("Test address seeds:", test_idx.size(0))
train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, train_idx, val_idx, test_idx, config)
model = HeteroTemporalAMLNet(config.hidden_channels, config.out_channels).to(device)
model = train_or_load_model(model, train_loader, val_loader, train_data, train_idx, config, device)
train_pred = score_loader_to_frame(
    model, train_loader, "train", config.train_max_step, builder.all_addr_ids_sorted, config, device
)
val_pred = score_loader_to_frame(
    model, val_loader, "validation", config.val_max_step, builder.all_addr_ids_sorted, config, device
)
test_pred = score_loader_to_frame(
    model, test_loader, "test", config.test_max_step, builder.all_addr_ids_sorted, config, device
)
pred_df = pd.concat([train_pred, val_pred, test_pred], ignore_index=True)
# 운영용 위험 점수 테이블
fraud_scores = build_fraud_scores(pred_df)
save_table(fraud_scores, config.output_dir / "scores" / "fraud_scores")
print("Saved fraud score table:", config.output_dir / "scores" / "fraud_scores.csv")
# 운영 지표 리포트
eval_summary, band_summary, time_summary = build_eval_reports(fraud_scores, config)
save_table(eval_summary, config.output_dir / "reports" / "model_eval_summary")
save_table(band_summary, config.output_dir / "reports" / "risk_band_summary")
save_table(time_summary, config.output_dir / "reports" / "time_step_summary")
print("Saved evaluation reports:", config.output_dir / "reports")
# address 매핑과 최종 address-address edge table
address_index_map = build_address_index_map(builder.all_addr_ids_sorted)
address_edge_table = export_address_edge_table(test_data, builder.all_addr_ids_sorted, config)
explain_base_summary = build_explain_base_summary(fraud_scores, address_index_map, address_edge_table)
save_table(address_index_map, config.output_dir / "explain" / "address_index_map")
save_table(address_edge_table, config.output_dir / "explain" / "address_addr_edges")
save_table(explain_base_summary, config.output_dir / "explain" / "explain_base_summary")
print("Saved minimal explanation base tables:", config.output_dir / "explain")
metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_name": config.model_name,
    "model_version": config.model_version,
    "train_max_step": config.train_max_step,
    "val_max_step": config.val_max_step,
    "test_max_step": config.test_max_step,
    "num_epochs": config.num_epochs,
    "hidden_channels": config.hidden_channels,
    "batch_size": config.batch_size,
    "num_train_seeds": int(train_idx.size(0)),
    "num_val_seeds": int(val_idx.size(0)),
    "num_test_seeds": int(test_idx.size(0)),
    "output_files": {
        "fraud_scores": str(config.output_dir / "scores" / "fraud_scores.csv"),
        "model_eval_summary": str(config.output_dir / "reports" / "model_eval_summary.csv"),
        "risk_band_summary": str(config.output_dir / "reports" / "risk_band_summary.csv"),
        "time_step_summary": str(config.output_dir / "reports" / "time_step_summary.csv"),
        "address_index_map": str(config.output_dir / "explain" / "address_index_map.csv"),
        "address_addr_edges": str(config.output_dir / "explain" / "address_addr_edges.csv"),
        "explain_base_summary": str(config.output_dir / "explain" / "explain_base_summary.csv"),
    },
}
with open(config.metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print("Saved metadata:", config.metadata_path)
# 모델 학습 및 결과 저장