import json
import os
import pickle
import shutil
import numpy as np
import pandas as pd
from easydict import EasyDict

# =============================== 配置 =============================== #
dataset_name = 'SD'
year = '2019'
# 这里的路径请确保正确
# base_raw_dir = f'/data/home/guogm/Code/BasicTS-master/datasets/raw_data/{dataset_name}'
base_raw_dir = f'/home/ggm/Code/BasicTS/datasets/raw_data/{dataset_name}'
data_file_path = f'{base_raw_dir}/{dataset_name}_{year}.h5'
graph_file_path = f'{base_raw_dir}/adj_{dataset_name}.npy'
meta_file_path = f'{base_raw_dir}/meta_{dataset_name}.csv'

# 【实验关键设置】
output_dir = f'/home/ggm/Code/BasicTS/datasets/{dataset_name}_Distance_baseline'
# output_dir = f'/home/ggm/Code/BasicTS/datasets/{dataset_name}_Distance'
target_channel = [0]
add_time_of_day = True
add_day_of_week = True
add_day_of_month = True
steps_per_day = 96
frequency = 1440 // steps_per_day
domain = 'traffic flow'
feature_description = [domain, 'time of day', 'day of week']
regular_settings = {
    'INPUT_LEN': 12,
    'OUTPUT_LEN': 12,
    'TRAIN_VAL_TEST_RATIO': [0.6, 0.2, 0.2],  # 训练集占 60%
    'NORM_EACH_CHANNEL': False,
    'RESCALE': True,
    'METRICS': ['MAE', 'RMSE', 'MAPE'],
    'NULL_VAL': 0.0,
    'INDUCTIVE_INFO': {'strategy': 'distance_based_expansion'},
    'FEW_SHOT_DAYS': 0.5
}

# unseen 节点数量
NUM_UNSEEN = 95

# 城市中心锚点
# 可以手动指定，也可以自动计算
USE_MANUAL_CENTER = False

MANUAL_CENTER_LAT = 32.7157
MANUAL_CENTER_LNG = -117.1611


# =============================== Distance-based Split =============================== #

def get_distance_split(df_meta):

    # =============================== 1. 确定城市中心 =============================== #

    if USE_MANUAL_CENTER:

        center_lat = MANUAL_CENTER_LAT
        center_lng = MANUAL_CENTER_LNG

    else:

        center_lat = df_meta["Lat"].mean()
        center_lng = df_meta["Lng"].mean()

    print(f"City Center: ({center_lat:.4f}, {center_lng:.4f})")

    # =============================== 2. 计算距离 =============================== #

    df_meta["distance"] = np.sqrt(
        (df_meta["Lat"] - center_lat) ** 2 +
        (df_meta["Lng"] - center_lng) ** 2
    )

    # =============================== 3. 选择最远的节点 =============================== #

    df_sorted = df_meta.sort_values(
        by="distance",
        ascending=False
    )

    unseen_indices = df_sorted.iloc[:NUM_UNSEEN].index.tolist()

    seen_indices = df_sorted.iloc[NUM_UNSEEN:].index.tolist()

    print(f"Seen Nodes: {len(seen_indices)}")
    print(f"Unseen Nodes: {len(unseen_indices)}")

    return seen_indices, unseen_indices

# =============================== 核心逻辑 =============================== #

def apply_inductive_logic(df_data):
    '''实现：训练期填充，验证/测试期真实'''
    df_meta = pd.read_csv(meta_file_path)

    # 1. 找到节点索引
    known_indices, target_indices = get_distance_split(df_meta)

    data = df_data.values.copy()  # [L, N]
    total_len = data.shape[0]

    # 2. 计算训练集的截止时间点 (基于 0.6 的比例)
    train_end_idx = int(total_len * regular_settings['TRAIN_VAL_TEST_RATIO'][0])
    train1_end_idx = train_end_idx - int(regular_settings['FEW_SHOT_DAYS'] * steps_per_day)

    print(f"Total time steps: {total_len}, Training ends at: {train_end_idx}, Train1 (Old only) ends at: {train1_end_idx}, Train2 (Few-shot, Old+New) length: {train_end_idx - train1_end_idx}")
    print(f"Target Nodes count: {len(target_indices)}")

    # 3. 执行填充逻辑
    # 仅针对训练集范围内的 Target Nodes 进行操作
    train1_slice  = data[:train1_end_idx, :]

    # 计算训练期内已知节点的全局均值 (按行计算每一时刻的均值)
    train1_known_mean = np.mean(train1_slice[:, known_indices], axis=1, keepdims=True)

    # 替换训练期内的目标节点数据为均值
    data[:train1_end_idx, target_indices] = train1_known_mean
    mask = np.ones_like(data)
    mask[:train1_end_idx, target_indices] = 0.0
    # 验证集和测试集范围 [train_end_idx : ] 保持 data 原样（即真实值）
    print("Padding applied to Training phase. Validation and Test phases remain real values.")

    return data, mask


def generate_inductive_indices():
    df_meta = pd.read_csv(meta_file_path)
    # 目标：找出所有【不属于】I5-N 的节点作为旧节点
    known_indices, target_indices = get_distance_split(df_meta)

    known_indices = np.array(known_indices)

    # 保存这个索引文件到数据集目录
    np.save(os.path.join(output_dir, 'known_node_indices.npy'), known_indices)
    print(f"Known nodes (Train): {len(known_indices)}, New nodes (Test): {len(target_indices)}")
    return known_indices


def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    df = pd.read_hdf(data_file_path).iloc[:96*90]
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data, df


def load_and_preprocess_data_baseline():
    df = pd.read_hdf(data_file_path).iloc[:96*90]
    # 应用我们的实验逻辑
    processed_values, mask = apply_inductive_logic(df)

    data = np.expand_dims(processed_values, axis=-1)
    mask = np.expand_dims(mask, axis=-1)  # [L, N, 1]
    data = data[..., target_channel]
    print(f'Final data shape: {data.shape}')
    return data, df, mask


# =============================== 工具函数 (保持原样) =============================== #

def add_temporal_features(data, df):
    '''Add time of day and day of week as features to the data.'''
    _, n, _ = data.shape
    feature_list = [data]

    if add_time_of_day:
        time_of_day = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = df.index.dayofweek / 7
        day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)

    if add_day_of_month:
        # numerical day_of_month
        day_of_month = (df.index.day - 1 ) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        day_of_month_tiled = np.tile(day_of_month, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_month_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x C
    return data_with_features

def save_data(data):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'data.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()

def save_mask(mask):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'mask.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=mask.shape)
    fp[:] = mask[:]
    fp.flush()

def save_graph():
    adj_mx = np.load(graph_file_path)
    with open(os.path.join(output_dir, 'adj_mx.pkl'), 'wb') as f:
        pickle.dump(adj_mx, f)


def save_meta_data():
    shutil.copyfile(meta_file_path, os.path.join(output_dir, 'meta_with_pois.csv'))


def save_description_with_mask(data,mask):
    description = {
        'name': dataset_name, 'shape': data.shape, 'num_nodes': data.shape[1], 'mask_shape':mask.shape,
        'feature_description': feature_description, 'regular_settings': regular_settings
    }
    with open(os.path.join(output_dir, 'desc.json'), 'w') as f:
        json.dump(description, f, indent=4)

def save_description(data):
    description = {
        'name': dataset_name, 'shape': data.shape, 'num_nodes': data.shape[1],
        'feature_description': feature_description, 'regular_settings': regular_settings
    }
    with open(os.path.join(output_dir, 'desc.json'), 'w') as f:
        json.dump(description, f, indent=4)


# =============================== 主程序 =============================== #

def baseline_main():
    # 1. 加载并处理数据（含填充逻辑）
    data_raw, df_index, mask = load_and_preprocess_data_baseline()
    # 2. 添加时间特征
    data_final = add_temporal_features(data_raw, df_index)
    # 3. 保存
    generate_inductive_indices()
    save_data(data_final)
    save_mask(mask)
    save_graph()
    save_meta_data()
    save_description_with_mask(data_final,mask)
    print(f"Inductive Baseline Data saved to {output_dir}")


def main():
    # Load and preprocess data
    data, df = load_and_preprocess_data()

    # Add temporal features
    data_with_features = add_temporal_features(data, df)

    generate_inductive_indices()

    # Save processed data
    save_data(data_with_features)

    # Copy and save adjacency matrix
    save_graph()

    # Copy and save meta data
    save_meta_data()

    # Save dataset description
    save_description(data_with_features)

if __name__ == '__main__':
    # generate_inductive_indices()
    # main()
    baseline_main()
