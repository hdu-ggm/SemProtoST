from typing import Dict, Optional
import torch
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
import numpy as np
from typing import Dict, Optional, Tuple, Union
from easytorch.utils import master_only
from basicts.metrics import masked_mae, semantic_consistency_loss
from tqdm import tqdm
import json
import os

class MetaRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for LargeST dataset.
    Handles traffic flow data alongside categorical and numerical metadata.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)



    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True,
                **kwargs) -> Dict:
        """
        Main forward pass for LargeST.
        """
        # 1. 归一化及键值对齐
        data = self.preprocessing(data)

        # 2. 将数据移动到运行设备
        # 流量数据 [B, L, N, C]
        history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])

        # 元数据 [B, N, D] 或 [N, D] (取决于 Dataset 是否在 batch 维度广播了)
        meta_cat = self.to_running_device(data['meta_categorical_indices'])
        meta_num = self.to_running_device(data['meta_numerical_values'])

        batch_size, length, num_nodes, _ = future_data.shape

        # 3. 特征选择 (如：只取流量，忽略时间 ID)
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # 推理模式下屏蔽未来特征的第一个维度（通常是真实值）
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # 4. 调用模型
        # 注意：这里我们修改了调用接口，传入了 meta_cat 和 meta_num
        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            meta_cat=meta_cat,
            meta_num=meta_num,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        # 6. 后处理 (反归一化)
        model_return = self.postprocessing(model_return)

        return model_return

