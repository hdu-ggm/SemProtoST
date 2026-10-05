import os
import torch
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .simple_tsf_dataset import TimeSeriesForecastingDataset
from baselines.SemProtoST.arch.STEncoder import LargeSTMetaModel


DEFAULT_CATEGORICAL_COLUMNS = ['District', 'County', 'Fwy', 'Type', 'Direction']
DEFAULT_NUMERICAL_COLUMNS = ['Residential', 'Workplace', 'Commercial', 'Transport', 'Lat', 'Lng', 'Lanes']
FULL_NUMERICAL_COLUMNS = [
    'Residential', 'Workplace', 'Commercial', 'Transport',
    'Lat', 'Lng', 'Lanes'
]

# Fields follow the grouping used in the ICDE metadata robustness experiments.
METADATA_GROUPS = {
    'geo': {
        'categorical': ['District', 'County'],
        'numerical': ['Lat', 'Lng'],
    },
    'road': {
        # Type is constant ("main") in all four datasets and therefore carries
        # no discriminative information. Keep it in the input tensor for
        # checkpoint compatibility, but do not treat it as a corruption field.
        'categorical': ['Fwy', 'Direction'],
        'numerical': ['Lanes'],
    },
    'poi': {
        'categorical': [],
        'numerical': ['Residential', 'Workplace', 'Commercial', 'Transport'],
    },
}

# 之前编写的元数据处理器，稍作调整以适应基础库路径
class LargeSTMetaProcessor:
    def __init__(self, csv_path, numerical_columns: Optional[Sequence[str]] = None):
        self.df = pd.read_csv(csv_path)
        self.cat_cols = list(DEFAULT_CATEGORICAL_COLUMNS)
        # Keep the current no-POI behavior by default. Robustness configs pass
        # FULL_NUMERICAL_COLUMNS explicitly when evaluating a clean checkpoint.
        self.num_cols = list(numerical_columns or DEFAULT_NUMERICAL_COLUMNS)
        missing_columns = [
            col for col in self.cat_cols + self.num_cols if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Metadata file {csv_path} is missing columns: {missing_columns}"
            )
        self.label_encoders = {col: LabelEncoder() for col in self.cat_cols}
        self.scaler = StandardScaler()

    def process(self):
        # 类别型处理
        cat_indices = []
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna('Unknown').astype(str)
            idx = self.label_encoders[col].fit_transform(self.df[col])
            cat_indices.append(idx)
        meta_cat = np.stack(cat_indices, axis=1)

        # 数值型处理
        for col in self.num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        meta_num = self.scaler.fit_transform(self.df[self.num_cols])

        # 记录每个类别的大小，供模型初始化 Embedding 使用
        self.cat_sizes = {col: len(self.label_encoders[col].classes_) for col in self.cat_cols}

        return torch.LongTensor(meta_cat), torch.FloatTensor(meta_num)


def _resolve_group_columns(
        group: str,
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str]) -> Tuple[List[int], List[int]]:
    """Resolve a semantic metadata group to tensor column indices."""
    group = group.lower()
    if group == 'all':
        requested_cat = [
            col for name in ('geo', 'road', 'poi')
            for col in METADATA_GROUPS[name]['categorical']
        ]
        requested_num = [
            col for name in ('geo', 'road', 'poi')
            for col in METADATA_GROUPS[name]['numerical']
        ]
    elif group in METADATA_GROUPS:
        requested_cat = METADATA_GROUPS[group]['categorical']
        requested_num = METADATA_GROUPS[group]['numerical']
    else:
        raise ValueError(
            f"Unknown metadata group '{group}'. Expected geo, road, poi, or all."
        )

    missing_cat = [col for col in requested_cat if col not in categorical_columns]
    missing_num = [col for col in requested_num if col not in numerical_columns]
    if missing_cat or missing_num:
        raise ValueError(
            f"Metadata group '{group}' cannot be corrupted because these fields "
            f"are unavailable: categorical={missing_cat}, numerical={missing_num}. "
            "Use the full metadata columns when evaluating the clean model."
        )

    cat_indices = [categorical_columns.index(col) for col in requested_cat]
    num_indices = [numerical_columns.index(col) for col in requested_num]
    return cat_indices, num_indices


def corrupt_unseen_metadata(
        meta_cat: torch.Tensor,
        meta_num: torch.Tensor,
        known_indices: Sequence[int],
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str],
        corruption: Mapping) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Corrupt metadata for a sampled subset of unseen nodes only.

    Missing values are replaced with statistics computed from D1 (known nodes):
    the mode for categorical fields and the mean for standardized numerical
    fields. Noise replaces categorical fields with a different valid D1 value
    and adds Gaussian noise to numerical fields in standardized space.
    """
    kind = str(corruption.get('kind', 'none')).lower()
    if kind not in {'missing', 'noise'}:
        raise ValueError(
            f"Unknown metadata corruption kind '{kind}'. Expected missing or noise."
        )

    group = str(corruption.get('group', 'all')).lower()
    rate = float(corruption.get('rate', 0.5 if kind == 'missing' else 1.0))
    seed = int(corruption.get('seed', 47))
    noise_std = float(corruption.get('noise_std', 0.3))
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Metadata corruption rate must be in [0, 1], got {rate}.")
    if noise_std < 0.0:
        raise ValueError(f"Metadata noise_std must be non-negative, got {noise_std}.")
    if meta_cat.ndim != 2 or meta_num.ndim != 2:
        raise ValueError("Metadata tensors must have shape [num_nodes, num_fields].")
    if meta_cat.shape[0] != meta_num.shape[0]:
        raise ValueError("Categorical and numerical metadata must have the same nodes.")

    known = np.asarray(known_indices, dtype=np.int64)
    if known.ndim != 1 or known.size == 0:
        raise ValueError("known_indices must be a non-empty one-dimensional array.")
    if known.min() < 0 or known.max() >= meta_cat.shape[0]:
        raise ValueError("known_indices contains an out-of-range node index.")

    all_nodes = np.arange(meta_cat.shape[0], dtype=np.int64)
    unseen = np.setdiff1d(all_nodes, known, assume_unique=False)
    sample_size = int(np.floor(rate * unseen.size))
    if rate > 0.0 and sample_size == 0 and unseen.size > 0:
        sample_size = 1

    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(unseen, size=sample_size, replace=False)
    ) if sample_size else np.empty(0, dtype=np.int64)

    corrupted_cat = meta_cat.clone()
    corrupted_num = meta_num.clone()
    cat_indices, num_indices = _resolve_group_columns(
        group, list(categorical_columns), list(numerical_columns)
    )
    if selected.size == 0:
        return corrupted_cat, corrupted_num, selected

    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    known_tensor = torch.as_tensor(known, dtype=torch.long)

    if kind == 'missing':
        for col_idx in cat_indices:
            fill_value = torch.mode(meta_cat[known_tensor, col_idx]).values
            corrupted_cat[selected_tensor, col_idx] = fill_value
        for col_idx in num_indices:
            fill_value = meta_num[known_tensor, col_idx].mean()
            corrupted_num[selected_tensor, col_idx] = fill_value
    else:
        for col_idx in cat_indices:
            pool = torch.unique(meta_cat[known_tensor, col_idx]).cpu().numpy()
            if pool.size <= 1:
                continue
            current = meta_cat[selected_tensor, col_idx].cpu().numpy()
            replacements = np.empty_like(current)
            for row_idx, current_value in enumerate(current):
                candidates = pool[pool != current_value]
                replacements[row_idx] = rng.choice(candidates if candidates.size else pool)
            corrupted_cat[selected_tensor, col_idx] = torch.as_tensor(
                replacements, dtype=corrupted_cat.dtype
            )
        if num_indices:
            noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=(selected.size, len(num_indices)),
            )
            noise_tensor = torch.as_tensor(noise, dtype=corrupted_num.dtype)
            for offset, col_idx in enumerate(num_indices):
                corrupted_num[selected_tensor, col_idx] += noise_tensor[:, offset]

    return corrupted_cat, corrupted_num, selected


class LargeSTDataset(TimeSeriesForecastingDataset):
    """
    LargeST Dataset class integrated with metadata for inductive spatio-temporal forecasting.
    Inherits from TimeSeriesForecastingDataset to support standardized data loading and splitting.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int,
                 overlap: bool = False, logger: logging.Logger = None,
                 metadata_numerical_columns: Optional[Sequence[str]] = None,
                 metadata_corruption: Optional[Dict] = None) -> None:
        # 1. 调用父类初始化，完成 data.dat 的加载和 train/valid/test 的分割
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap, logger)

        # 2. 定位并处理元数据
        self.meta_file_path = f'datasets/{dataset_name}/meta_with_pois.csv'
        if not os.path.exists(self.meta_file_path):
            raise FileNotFoundError(
                f"Metadata file not found at {self.meta_file_path}. Inductive features require meta.csv.")

        self.processor = LargeSTMetaProcessor(
            self.meta_file_path,
            numerical_columns=metadata_numerical_columns,
        )
        self.meta_cat, self.meta_num = self.processor.process()

        # 暴露类别大小给模型
        self.cat_sizes = self.processor.cat_sizes

        # 1. 加载我们生成的旧节点索引
        idx_path = f'datasets/{dataset_name}/known_node_indices.npy'
        self.known_indices = np.load(idx_path)

        # Robustness corruptions are deliberately test-only. Train and valid
        # always receive clean metadata, so one clean checkpoint can be reused.
        self.metadata_corruption = metadata_corruption
        self.metadata_corruption_indices = np.empty(0, dtype=np.int64)
        if self.mode == 'test' and metadata_corruption:
            self.meta_cat, self.meta_num, self.metadata_corruption_indices = \
                corrupt_unseen_metadata(
                    self.meta_cat,
                    self.meta_num,
                    self.known_indices,
                    self.processor.cat_cols,
                    self.processor.num_cols,
                    metadata_corruption,
                )
            kind = metadata_corruption.get('kind', 'none')
            group = metadata_corruption.get('group', 'all')
            rate = metadata_corruption.get('rate', 0.5)
            message = (
                f">>> [Metadata robustness] kind={kind}, group={group}, "
                f"rate={rate}, seed={metadata_corruption.get('seed', 47)}, "
                f"corrupted unseen nodes={len(self.metadata_corruption_indices)}"
            )
            print(message)
            if self.logger is not None:
                self.logger.info(message)

        # 2. 根据模式决定“可见”的节点
        if self.mode == 'train':
            # 训练模式：物理删除新节点，只保留旧节点
            # 此时 self.data 的维度会从 [L, N_all, C] 变成 [L, N_old, C]
            self.data = self.get_zero_data(self.data)
            print("---------------------------------------------")
            print(self.data.shape)
            self.meta_cat = self.meta_cat[self.known_indices]
            self.meta_num = self.meta_num[self.known_indices]
            print(f">>> [Dataset] TRAIN mode: Using ONLY {len(self.known_indices)} known nodes.")
        else:
            # 验证和测试模式：保留所有节点（旧 + 新）且为真实值
            print(f">>> [Dataset] {self.mode.upper()} mode: Using ALL {self.data.shape[1]} nodes.")
            print(self.data.shape)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample, extending the base class with metadata tensors.
        """
        # 调用父类的 __getitem__ 获取 {'inputs': ..., 'target': ...}
        sample = super().__getitem__(index)

        # 为了兼容 PyTorch 默认的输入格式，通常需要增加特征维度 (C=1)
        # 将 [L, N] 转换为 [L, N, 1]
        # inputs = torch.from_numpy(sample['inputs']).unsqueeze(-1)
        # target = torch.from_numpy(sample['target']).unsqueeze(-1)
        # print(inputs.shape)
        # 注入元数据
        # meta_categorical_indices: [num_nodes, 5]
        # meta_numerical_values: [num_nodes, 3]
        return {
            'inputs': sample['inputs'],
            'target': sample['target'],
            'meta_categorical_indices': self.meta_cat,
            'meta_numerical_values': self.meta_num
        }

    def get_metadata_info(self):
        """
        Returns categorical sizes for model's embedding layer configuration.
        """
        return self.cat_sizes

    def get_zero_data(self, data):
        # data L N C
        total_len = data.shape[0]
        zero_train_end_idx = int(total_len - 0.5 * 96)
        data = data[:zero_train_end_idx, self.known_indices, :]
        return data
