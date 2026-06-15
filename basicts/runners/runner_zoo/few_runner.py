from typing import Dict, Optional
import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
import numpy as np
from typing import Dict, Optional, Tuple, Union
from easytorch.utils import master_only
from basicts.metrics import masked_mae, semantic_consistency_loss, masked_mse
from easytorch.core.checkpoint import load_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import json
import time
import os
from torch import nn, optim
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             is_master, master_only, set_env)


class FewRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for LargeST dataset.
    Handles traffic flow data alongside categorical and numerical metadata.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.teacher_model = self.build_model(cfg)
        ckpt_path = cfg['TRAIN.FINETUNE_FROM']
        strict = cfg.get('TRAIN.FINETUNE_STRICT_LOAD', True)
        try:
            print(self.ckpt_save_dir)
            print(ckpt_path)
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            if isinstance(self.teacher_model, DDP):
                self.teacher_model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.teacher_model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError) as e:
            raise OSError('Ckpt file does not exist') from e

        dataset_name = cfg['DATASET']['PARAM'].get('dataset_name', 'SD_Inductive')
        # 1. 加载旧节点索引 (已知节点)
        idx_path = f'datasets/{dataset_name}/known_node_indices.npy'
        self.known_node_indices = np.load(idx_path)

        # 2. 获取总节点数并推导“新节点索引”
        # 假设总节点数在配置里，或者你已经知道是 716
        num_nodes = cfg['MODEL']['PARAM'].get('num_nodes', 716)
        all_node_indices = np.arange(num_nodes)

        # 使用 np.setdiff1d 找出不在 known_node_indices 里的 ID
        self.new_node_indices = np.setdiff1d(all_node_indices, self.known_node_indices)

        self.logger.info(
            f">>> Inductive Setup: {len(self.known_node_indices)} Known, {len(self.new_node_indices)} New.")
        self.lambda_ortho = 1000
        self.lambda_semantic = 100
        self.lambda_distill = cfg['TRAIN.DISTILL']

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        self.register_epoch_meter('train/loss_ortho', 'train', '{:.4f}')
        self.register_epoch_meter('train/loss_semantic', 'train', '{:.4f}')

    def build_optim(self, optim_cfg: Dict, model: nn.Module) -> optim.Optimizer:
        if isinstance(optim_cfg['TYPE'], type):
            optim_type = optim_cfg['TYPE']
        else:
            if hasattr(optim, optim_cfg['TYPE']):
                optim_type = getattr(optim, optim_cfg['TYPE'])
            else:
                # 兼容 basicts 或其他自定义优化器
                import basicts.runners.optim.optimizers as basicts_optim
                optim_type = getattr(basicts_optim, optim_cfg['TYPE'])

            # 2. 获取基础全局参数 (lr, weight_decay)
        base_optim_param = optim_cfg['PARAM'].copy()

        # 3. 处理参数组
        if "PARAM_GROUPS" in optim_cfg:
            param_groups = []
            memo = set()  # 用来记录已经分配过的参数，防止重复添加

            # 遍历配置中的每一个组
            for group_info in optim_cfg["PARAM_GROUPS"]:
                group_config = group_info.copy()
                module_name = group_config.pop("name")  # 提取模块名，如 'encoder'

                if hasattr(model, module_name):
                    module = getattr(model, module_name)
                    # 筛选该模块中需要梯度的参数
                    params = [p for p in module.parameters() if p.requires_grad]

                    if len(params) > 0:
                        # 组装该组的参数：基础全局参数 + 该组特定的覆盖参数(如特定的lr)
                        new_group = {"params": params, **base_optim_param}
                        new_group.update(group_config)

                        param_groups.append(new_group)

                        # 将参数加入已处理集合
                        for p in params:
                            memo.add(p)
                else:
                    print(f"Warning: Model has no module named {module_name}")

            # 4. 关键：处理模型中剩余的、未在 PARAM_GROUPS 中定义的参数
            remaining_params = [p for p in model.parameters() if p.requires_grad and p not in memo]
            if remaining_params:
                # 剩余参数使用基础全局参数配置
                param_groups.append({"params": remaining_params, **base_optim_param})

            optimizer = optim_type(param_groups)
        else:
            # 如果没有定义 PARAM_GROUPS，按传统方式初始化所有参数
            optimizer = optim_type(filter(lambda p: p.requires_grad, model.parameters()), **base_optim_param)

        return optimizer


    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self.register_epoch_meter('val/loss_ortho', 'val', '{:.4f}')
        self.register_epoch_meter('val/loss_semantic', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):
        super().init_test(cfg)
        self.register_epoch_meter('test/loss_ortho', 'test', '{:.4f}')
        self.register_epoch_meter('test/loss_semantic', 'test', '{:.4f}')

    def preprocessing(self, input_data: Dict) -> Dict:
        """
        Preprocess LargeST data.
        Maps custom keys to base keys ('inputs', 'target') for scaling.
        """
        # 将 LargeSTDataset 的键映射到基类预处理逻辑所需的键
        # input_data['inputs'] = input_data['traffic_history']
        # input_data['target'] = input_data['ground_truth']

        # 调用父类的归一化逻辑 (使用 self.scaler)
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
        model_return, loss_ortho, attn, meta_node = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            meta_cat=meta_cat,
            meta_num=meta_num,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        with torch.no_grad():
            teacher_return, _, _, _ = self.teacher_model(
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
            model_return = {'prediction': model_return, 'teacher_prediction': teacher_return,'loss_ortho': loss_ortho, 'attn': attn, 'meta_node': meta_node}

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
        # rescale data
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['teacher_prediction'] = self.scaler.inverse_transform(input_data['teacher_prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        # subset forecasting
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]
            input_data['teacher_prediction'] = input_data['teacher_prediction'][:, :, self.target_time_series, :]

        return input_data


    def train(self, cfg: Dict) -> None:
        """Train model.
            The start point of training process.

        Train process:
        [init_training]
        for in train_epoch
            [on_epoch_start]
            for in train iters
                [train_iters]
            [on_epoch_end] ------> Epoch Val: val every n epoch
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
        [on_training_end]

        Args:
            cfg (Dict): config
        """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)

        epoch_index = 0
        # training loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1

            # early stopping check
            if self.check_early_stopping():
                break

            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # Freeze
            for name, param in self.model.named_parameters():

                # ✅ 只允许这两个模块更新
                if ("spatial_bank" in name) or ("regression_layer" in name) or ("temporal_bank" in name) or ("encoder" in name)\
                        or ("num_projector" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # tqdm process bar
            data_loader = tqdm(self.train_data_loader) if get_local_rank() == 0 \
                                                        else self.train_data_loader

            # data loop
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train/time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end(cfg=cfg, train_epoch=epoch_index + 1)

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['teacher_prediction'] = forward_return['teacher_prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        # loss = self.metric_forward(self.loss, forward_return)

        prediction = forward_return['prediction']  # [B, 12, N, 1]
        teacher_prediction = forward_return['teacher_prediction']  # [B, 12, N, 1]
        # self.known_node_indices 是你在数据准备阶段保存的老节点索引
        student_known = prediction[:, :, self.known_node_indices, :]
        teacher_known = teacher_prediction[:, :, self.known_node_indices, :]
        loss_distill = masked_mse(student_known, teacher_known)

        attn = forward_return['attn']  # [B, N, K] 空间银行的注意力权重
        meta_node = forward_return['meta_node']  # [B, N, 7] POI 数据

        B, L, N, _ = prediction.shape

        # ==================== 【虚拟归纳手术开始】 ====================
        # 1. 随机生成节点掩码
        perm = torch.randperm(N).to(prediction.device)
        cutoff = int(0.7 * N)  # 80% 节点作为“已知点”

        seen_idx = perm[:cutoff]
        unseen_idx = perm[cutoff:]

        loss_semantic_unseen = semantic_consistency_loss(attn, meta_node, self.new_node_indices, self.known_node_indices)

        loss_ortho = forward_return['loss_ortho']

        forward_return['prediction'] = forward_return['prediction'][:, :, self.new_node_indices, :]
        forward_return['target'] = forward_return['target'][:, :, self.new_node_indices, :]
        loss = self.metric_forward(self.loss, forward_return)

        total_loss = loss + self.lambda_ortho*loss_ortho + self.lambda_semantic * loss_semantic_unseen + loss_distill * self.lambda_distill
        # total_loss = loss  + loss_distill * self.lambda_distill
        self.update_epoch_meter('train/loss', total_loss.item())
        self.update_epoch_meter('train/loss_ortho', (self.lambda_ortho*loss_ortho).item())
        self.update_epoch_meter('train/loss_semantic', (self.lambda_semantic * loss_semantic_unseen).item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())

        return total_loss


    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        loss_ortho = forward_return['loss_ortho']
        self.update_epoch_meter('val/loss', loss.item())
        self.update_epoch_meter('val/loss_ortho', (self.lambda_ortho*loss_ortho).item())

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
            loss_ortho = forward_return['loss_ortho']
            self.update_epoch_meter('test/loss', loss.item())
            self.update_epoch_meter('test/loss_ortho', (self.lambda_ortho * loss_ortho).item())

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