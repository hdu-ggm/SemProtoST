from typing import Dict, Optional
import torch
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
import numpy as np
from typing import Dict, Optional, Tuple, Union
from easytorch.utils import master_only
from tqdm import tqdm
import json
import os

class VQRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for LargeST dataset.
    Handles traffic flow data alongside categorical and numerical metadata.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.lambda_pattern = 1

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        self.register_epoch_meter('train/loss_pattern', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self.register_epoch_meter('val/loss_pattern', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):
        super().init_test(cfg)
        self.register_epoch_meter('test/loss_pattern', 'test', '{:.4f}')

    def preprocessing(self, input_data: Dict) -> Dict:


        input_data = super().preprocessing(input_data)
        return input_data

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
        model_return, loss_pattern = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            meta_cat=meta_cat,
            meta_num=meta_num,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        # 5. 封装结果用于 Loss 计算和评估
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return, 'loss_pattern': loss_pattern}

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # 6. 后处理 (反归一化)
        model_return = self.postprocessing(model_return)

        return model_return

    def postprocessing(self, input_data: Dict) -> Dict:
        """
        Postprocess and ensure keys are consistent with evaluation metrics.
        """
        # 调用父类反归一化逻辑
        input_data = super().postprocessing(input_data)

        # 如果需要，可以在这里把 'inputs' 重新映射回 'traffic_history' 等
        # 但通常基础库的 Evaluator 只认 'prediction' 和 'target'
        return input_data

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        loss = self.metric_forward(self.loss, forward_return)


        loss_pattern = forward_return['loss_pattern']

        total_loss = loss + self.lambda_pattern*loss_pattern
        self.update_epoch_meter('train/loss', total_loss.item())
        self.update_epoch_meter('train/loss_pattern', (self.lambda_pattern*loss_pattern).item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return total_loss


    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        loss_pattern = forward_return['loss_pattern']
        self.update_epoch_meter('val/loss', loss.item())
        self.update_epoch_meter('val/loss_pattern', (self.lambda_pattern*loss_pattern).item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item())


    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.

        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        prediction, target, inputs = [], [], []

        for data in tqdm(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            loss_pattern = forward_return['loss_pattern']
            self.update_epoch_meter('test/loss', loss.item())
            self.update_epoch_meter('test/loss_pattern', (self.lambda_pattern * loss_pattern).item())

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        metrics_results = self.compute_evaluation_metrics(returns_all)

        # save
        if save_results:
            # save returns_all to self.ckpt_save_dir/test_results.npz
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/test_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        return returns_all

