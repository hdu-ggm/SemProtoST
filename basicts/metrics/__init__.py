from .corr import masked_corr
from .mae import masked_mae
from .mape import masked_mape
from .mse import masked_mse
from .r_square import masked_r2
from .rmse import masked_rmse
from .smape import masked_smape
from .wape import masked_wape
from .huber import masked_huber
from .semantic_loss import semantic_consistency_loss
from .semantic_loss import semantic_consistency_loss_sample

ALL_METRICS = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape,
            'SMAPE': masked_smape,
            'R2': masked_r2,
            'CORR': masked_corr,
            'HUBER': masked_huber,
            'Semantic Loss': semantic_consistency_loss,
            'Semantic Loss Sample': semantic_consistency_loss_sample,
            }

__all__ = [
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'masked_huber',
    'semantic_consistency_loss',
    'semantic_consistency_loss_sample',
    'ALL_METRICS',
]