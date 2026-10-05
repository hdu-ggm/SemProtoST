import os
import torch
import pandas as pd
import numpy as np
import logging
from typing import List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .simple_tsf_dataset import TimeSeriesForecastingDataset
from baselines.SemProtoST.arch.STEncoder import LargeSTMetaModel

# 之前编写的元数据处理器，稍作调整以适应基础库路径
class LargeSTMetaProcessor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.cat_cols = ['District', 'County', 'Fwy', 'Type', 'Direction']
        self.num_cols = ['Lat', 'Lng', 'Lanes']
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
        self.df['Lanes'] = self.df['Lanes'].fillna(self.df['Lanes'].mean())
        meta_num = self.scaler.fit_transform(self.df[self.num_cols])

        # 记录每个类别的大小，供模型初始化 Embedding 使用
        self.cat_sizes = {col: len(self.label_encoders[col].classes_) for col in self.cat_cols}

        return torch.LongTensor(meta_cat), torch.FloatTensor(meta_num)


class MetaDataset(TimeSeriesForecastingDataset):
    """
    LargeST Dataset class integrated with metadata for inductive spatio-temporal forecasting.
    Inherits from TimeSeriesForecastingDataset to support standardized data loading and splitting.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int,
                 overlap: bool = False, logger: logging.Logger = None) -> None:
        # 1. 调用父类初始化，完成 data.dat 的加载和 train/valid/test 的分割
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap, logger)

        # 2. 定位并处理元数据
        self.meta_file_path = f'datasets/{dataset_name}/meta_with_pois.csv'
        if not os.path.exists(self.meta_file_path):
            raise FileNotFoundError(
                f"Metadata file not found at {self.meta_file_path}. Inductive features require meta.csv.")

        self.processor = LargeSTMetaProcessor(self.meta_file_path)
        self.meta_cat, self.meta_num = self.processor.process()

        # 暴露类别大小给模型
        self.cat_sizes = self.processor.cat_sizes



    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample, extending the base class with metadata tensors.
        """
        # 调用父类的 __getitem__ 获取 {'inputs': ..., 'target': ...}
        sample = super().__getitem__(index)


        return {
            'inputs': sample['inputs'],
            'target': sample['target'],
            'meta_categorical_indices': self.meta_cat,
            'meta_numerical_values': self.meta_num
        }

