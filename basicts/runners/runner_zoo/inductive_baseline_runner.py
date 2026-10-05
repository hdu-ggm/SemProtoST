from typing import Dict, Optional
import torch
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
import numpy as np
from typing import Dict, Optional, Tuple, Union
from basicts.metrics.mae import masked_mae

class InductiveBaselineRunner(SimpleTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # 1. 加载旧节点索引 (已知节点)
        idx_path = f"datasets/{cfg['DATASET']['NAME']}/known_node_indices.npy"
        self.known_node_indices = np.load(idx_path)

        # 2. 获取总节点数并推导“新节点索引”
        # 假设总节点数在配置里，或者你已经知道是 716
        print(cfg['MODEL']['PARAM'])
        num_nodes = cfg['MODEL']['PARAM'].get('num_nodes', 716)
        all_node_indices = np.arange(num_nodes)

        # 使用 np.setdiff1d 找出不在 known_node_indices 里的 ID
        self.new_node_indices = np.setdiff1d(all_node_indices, self.known_node_indices)

        self.logger.info(
            f">>> Inductive Setup: {len(self.known_node_indices)} Known, {len(self.new_node_indices)} New.")



    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        mask = self.to_running_device(data['mask']) if 'mask' in data else None
        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)
        if 'mask' not in model_return:
            model_return['mask'] = mask

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return


    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        prediction = forward_return['prediction']
        target = forward_return['target']
        mask = forward_return['mask'] # 来自数据的 mask

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        if mask is not None:
            # 确保维度对齐
            if mask.dim() < prediction.dim():
                mask = mask.unsqueeze(-1)

            # 1. 计算掩码后的标量 Loss
            diff = torch.abs(prediction - target)
            loss = torch.sum(diff * mask) / (torch.sum(mask) + 1e-9)

        else:
            loss = self.metric_forward(self.loss, forward_return)


        self.update_epoch_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return loss



    def compute_evaluation_metrics(self, returns_all: Dict):
        """
        分拆计算已知节点和新增节点的指标
        returns_all: {'prediction': [B, L, N, C], 'target': [B, L, N, C], ...}
        """
        metrics_results = {}

        # 定义三组评估对象
        # None 代表全量计算
        node_groups = {
            'Overall': None,
            'Known_Nodes': self.known_node_indices,
            'New_Nodes': self.new_node_indices
        }

        # 1. 针对特定的预测步长 (Horizon) 进行详细评估
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i + 1}'] = {}

            for group_name, indices in node_groups.items():
                # 提取预测值和真实值 [Batch, Nodes, Channel]
                pred = returns_all['prediction'][:, i, :, :]
                real = returns_all['target'][:, i, :, :]

                # 执行空间维度切片 (如果不是 Overall)
                if indices is not None:
                    pred = pred[:, indices, :]
                    real = real[:, indices, :]

                group_metrics = {}
                metric_repr = f"[{group_name}]"

                for metric_name, metric_func in self.metrics.items():
                    if metric_name.lower() == 'mase': continue
                    # 计算该组的 MAE/RMSE 等
                    metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                    group_metrics[metric_name] = metric_item.item()
                    metric_repr += f" {metric_name}: {metric_item.item():.4f}"

                metrics_results[f'horizon_{i + 1}'][group_name] = group_metrics
                # 只打印非 Overall 的细分项，或者你也可以全打
                if group_name != 'Overall':
                    self.logger.info(f"Horizon {i + 1} {metric_repr}")

        # 2. 计算全局平均汇总指标 (Summary)
        metrics_results['summary'] = {}
        self.logger.info("*" * 20 + " Final Summary " + "*" * 20)

        for group_name, indices in node_groups.items():
            pred = returns_all['prediction']
            real = returns_all['target']

            if indices is not None:
                # 在第 2 维（N 维度）进行切片
                pred = pred[:, :, indices, :]
                real = real[:, :, indices, :]

            summary_metrics = {}
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                summary_metrics[metric_name] = metric_item.item()
                # 更新主指标到 Meter（用于保存 Best Model）
                if group_name == 'Overall':
                    self.update_epoch_meter(f'test/{metric_name}', metric_item.item())

            metrics_results['summary'][group_name] = summary_metrics
            self.logger.info(
                f"Summary {group_name} -> MAE: {summary_metrics['MAE']:.4f}, MAPE: {summary_metrics['MAPE']:.4f}, RMSE: {summary_metrics['RMSE']:.4f}")

        return metrics_results