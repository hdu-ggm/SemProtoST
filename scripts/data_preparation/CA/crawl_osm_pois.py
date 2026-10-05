import osmnx as ox
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# =============================== 1. 配置 =============================== #
INPUT_CSV = '/data/home/guogm/Code/BasicTS-master/datasets/CA/meta.csv'  # 你的原始元数据文件
OUTPUT_CSV = '/data/home/guogm/Code/BasicTS-master/datasets/CA/meta_with_pois.csv'  # 最终保存的文件
RADIUS = 500  # 抓取半径（米）

# 优化后的标签：去掉了 building: True 以防超时，改用精准标签
ST_TAGS = {
    "amenity": ["school", "university", "college", "hospital", "clinic",
                "restaurant", "cafe", "fast_food", "bar", "bank", "parking"],
    "shop": True,
    "office": True,
    "leisure": ["park", "stadium", "pitch"],
    "tourism": ["museum", "gallery", "theme_park", "attraction"],
    "railway": ["station", "subway_entrance"],
    "highway": ["bus_stop"],
    "landuse": ["residential", "commercial", "industrial"]
}


# =============================== 2. 核心抓取函数 =============================== #
def get_node_poi_features(lat, lng, radius):
    """抓取并统计POI，返回四个大类的计数字典"""
    stats = {"Residential": 0, "Workplace": 0, "Commercial": 0, "Transport": 0}
    try:
        ox.settings.timeout = 60  # 缩短超时时间，坏点直接跳过
        pois = ox.features_from_point((lat, lng), tags=ST_TAGS, dist=radius)

        if pois.empty:
            return stats

        cols = pois.columns
        # 1. Residential
        if 'landuse' in cols:
            stats["Residential"] += pois['landuse'].isin(['residential']).sum()
        if 'building' in cols:
            stats["Residential"] += pois['building'].isin(['apartments', 'residential']).sum()
        # 2. Workplace
        if 'office' in cols:
            stats["Workplace"] += pois['office'].notnull().sum()
        if 'amenity' in cols:
            stats["Workplace"] += pois['amenity'].isin(['school', 'university', 'college']).sum()
        # 3. Commercial
        if 'shop' in cols:
            stats["Commercial"] += pois['shop'].notnull().sum()
        if 'amenity' in cols:
            stats["Commercial"] += pois['amenity'].isin(['restaurant', 'cafe', 'fast_food', 'bar']).sum()
        # 4. Transport / Public
        if 'railway' in cols:
            stats["Transport"] += (pois['railway'] == 'station').sum()
        if 'amenity' in cols:
            stats["Transport"] += pois['amenity'].isin(['hospital', 'parking']).sum()
        if 'leisure' in cols:
            stats["Transport"] += pois['leisure'].isin(['park', 'stadium']).sum()

        return stats
    except Exception:
        # 网络错误或该点无数据，返回全0
        return stats


# =============================== 3. 主循环处理 =============================== #
def main():
    # 1. 加载原始数据
    if not os.path.exists(INPUT_CSV):
        print(f"错误：找不到文件 {INPUT_CSV}")
        return

    df_meta = pd.read_csv(INPUT_CSV)

    # 2. 检查进度：如果输出文件已存在，则读取它
    if os.path.exists(OUTPUT_CSV):
        df_processed = pd.read_csv(OUTPUT_CSV)
        start_row = len(df_processed)
        # 将已有的数据加载进列表，确保最终保存时包含旧数据
        enriched_data = df_processed.to_dict('records')
        print(f"检测到已存在输出文件，从第 {start_row} 行（索引 {start_row}）开始续传...")
    else:
        start_row = 0
        enriched_data = []
        print("未检测到处理记录，从头开始抓取...")

    # 3. 只处理未完成的部分 (使用 iloc 切片)
    # df_meta.iloc[start_row:] 会从 start_row 开始直到最后
    df_to_process = df_meta.iloc[start_row:]

    for i, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        try:
            # 获取 POI 特征
            poi_stats = get_node_poi_features(row['Lat'], row['Lng'], RADIUS)

            # 合并数据
            combined_row = {**row.to_dict(), **poi_stats}
            enriched_data.append(combined_row)

            # 每 20 个点自动保存一次
            # 注意：这里的 i 是原始 df_meta 的索引
            if (i + 1) % 20 == 0:
                pd.DataFrame(enriched_data).to_csv(OUTPUT_CSV, index=False)

        except Exception as e:
            # 如果中间报错，先保存当前已抓取的，方便下次再续传
            pd.DataFrame(enriched_data).to_csv(OUTPUT_CSV, index=False)
            print(f"\n在处理第 {i} 行时发生错误: {e}")
            print("进度已保存，程序退出。")
            return

    # 最终保存
    final_df = pd.DataFrame(enriched_data)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("-" * 30)
    print(f"所有任务处理完成！")
    print(f"保存路径: {OUTPUT_CSV}")
    print("-" * 30)

if __name__ == "__main__":
    main()