"""Environment-driven configuration for metadata robustness evaluation.

The helper is inactive unless ``SEMPROTO_META_KIND`` is set. This preserves
existing training and ablation configs while allowing the same zero-shot
checkpoint to be evaluated under multiple metadata corruptions.
"""

import os


FULL_METADATA_NUMERICAL_COLUMNS = [
    'Residential', 'Workplace', 'Commercial', 'Transport',
    'Lat', 'Lng', 'Lanes'
]


def configure_metadata_robustness_from_env(cfg):
    """Apply test-time metadata corruption settings from environment variables."""
    kind = os.getenv('SEMPROTO_META_KIND')
    if not kind:
        return cfg

    kind = kind.lower()
    if kind not in {'missing', 'noise'}:
        raise ValueError(
            "SEMPROTO_META_KIND must be either 'missing' or 'noise'."
        )

    group = os.getenv('SEMPROTO_META_GROUP', 'all').lower()
    if group not in {'geo', 'road', 'poi', 'all'}:
        raise ValueError(
            "SEMPROTO_META_GROUP must be geo, road, poi, or all."
        )

    default_rate = '0.5' if kind == 'missing' else '1.0'
    cfg.DATASET.PARAM.metadata_numerical_columns = \
        list(FULL_METADATA_NUMERICAL_COLUMNS)
    cfg.DATASET.PARAM.metadata_corruption = {
        'kind': kind,
        'group': group,
        'rate': float(os.getenv('SEMPROTO_META_RATE', default_rate)),
        'noise_std': float(os.getenv('SEMPROTO_META_NOISE_STD', '0.3')),
        'seed': int(os.getenv('SEMPROTO_META_SEED', '47')),
    }

    # The full model checkpoint was trained with seven numerical fields.
    cfg.MODEL.PARAM['number_size'] = len(FULL_METADATA_NUMERICAL_COLUMNS)
    corruption = cfg.DATASET.PARAM.metadata_corruption
    run_name_parts = [
        kind,
        group,
        f"rate_{corruption['rate']:g}",
    ]
    if kind == 'noise':
        run_name_parts.append(f"std_{corruption['noise_std']:g}")
    run_name_parts.append(f"seed_{corruption['seed']}")

    # Evaluation writes fixed filenames (test_metrics.json/test_results.npz).
    # Give every corruption run its own directory to prevent overwriting.
    model_ckpt_root = os.path.dirname(cfg.TRAIN.CKPT_SAVE_DIR)
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        model_ckpt_root,
        f"{cfg.DATASET.NAME}_metadata_robustness",
        '_'.join(run_name_parts),
    )
    cfg.DESCRIPTION = (
        f"Metadata robustness: {kind}/{group}, "
        f"rate={corruption['rate']}, seed={corruption['seed']}"
    )
    return cfg
