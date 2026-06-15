import osmnx as ox
import pandas as pd
from tqdm import tqdm

# 定义我们关心的 POI 类别
tags = {
    'amenity': ['restaurant', 'hospital', 'school', 'university'],
    'shop': ['mall', 'supermarket'],
    'tourism': ['theme_park', 'attraction'],
    'landuse': ['industrial', 'residential', 'commercial'],
    'public_transport': ['station', 'stop_position']
}


def get_node_poi_features(lat, lng, radius=1000):
    try:
        # 在指定半径内抓取 POI
        pois = ox.features_from_point((lat, lng), tags, dist=radius)
        if pois.empty:
            return {}

        # 统计每一类的数量（例如：该区域内有多少个餐馆）
        stats = {}
        if 'amenity' in pois.columns:
            stats['count_amenity'] = pois['amenity'].count()
        if 'shop' in pois.columns:
            stats['count_shop'] = pois['shop'].count()
        # 也可以计算 POI 的多样性（熵）
        stats['poi_diversity'] = len(pois.columns)
        return stats
    except:
        return {}


# # 加载你的 meta.csv
# meta_df = pd.read_csv('meta.csv')
# poi_list = []
#
# print("Fetching OSM data for sensors...")
# for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
#     feat = get_node_poi_features(row['Lat'], row['Lng'])
#     feat['node_id'] = row['ID']
#     poi_list.append(feat)
#
# poi_df = pd.DataFrame(poi_list).fillna(0)
# poi_df.to_csv('node_pois.csv', index=False)

if __name__ == '__main__':
    get_node_poi_features(32.544463, -117.032486, 500)