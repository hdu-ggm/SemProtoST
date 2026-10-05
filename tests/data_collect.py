import requests
import os
import time
from datetime import datetime, timedelta

# https://new.portal.its.pdx.edu/highways/api/freewaydata/
# ?start_date=2025-12-29&end_date=2025-12-30&days_of_week=1&days_of_week=2&days_of_week=3&days_of_week=4&days_of_week=5&days_of_week=6&days_of_week=7
# &format=csv&highway_id=3&resolution=00%3A15%3A00
def data_down():
    # --- 1. 配置参数 ---
    # 在这里填入你需要下载的所有 HIGHWAY_ID
    HIGHWAY_IDS = ["619","627"]  # 示例：你可以按需添加更多

    BASE_URL = "https://new.portal.its.pdx.edu/highways/api/freewaydata/"

    # 设置时间范围（推荐 2 个月，如 2024-09-01 到 2024-11-01）
    START_DATE = datetime(2024, 3, 1)
    END_DATE = datetime(2024, 9, 1)



    # 总存储根目录
    ROOT_SAVE_DIR = "/data/home/guogm/Code/data"

    # --- 2. 构造通用参数模板 ---
    params_template = {
        'days_of_week': [1, 2, 3, 4, 5, 6, 7],
        'format': 'csv',
        'resolution': '00:15:00'
    }

    # --- 3. 执行多路段下载逻辑 ---
    for hwy_id in HIGHWAY_IDS:
        print(f"\n" + "=" * 50)
        print(f"正在处理 HIGHWAY_ID: {hwy_id}")
        print("=" * 50)

        # 为每个高速路段创建独立子文件夹
        hwy_dir = os.path.join(ROOT_SAVE_DIR, f"highway_{hwy_id}")
        if not os.path.exists(hwy_dir):
            os.makedirs(hwy_dir)

        current_start = START_DATE

        while current_start < END_DATE:
            current_end = current_start + timedelta(days=7)  # 按周分块
            if current_end > END_DATE:
                current_end = END_DATE

            s_str = current_start.strftime('%Y-%m-%d')
            e_str = current_end.strftime('%Y-%m-%d')

            # 构造当前请求参数
            current_params = params_template.copy()
            current_params['highway_id'] = hwy_id
            current_params['start_date'] = s_str
            current_params['end_date'] = e_str

            filename = f"hwy_{hwy_id}_{s_str}_to_{e_str}.csv"
            save_path = os.path.join(hwy_dir, filename)

            # 检查是否已经下载过，避免重复下载
            if os.path.exists(save_path):
                print(f"跳过已存在文件: {filename}")
                current_start = current_end
                continue

            print(f"正在下载: {s_str} -> {e_str} ...", end=" ", flush=True)

            try:
                # 请求数据
                response = requests.get(BASE_URL, params=current_params, timeout=300, proxies={"http": None, "https": None})

                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    print("成功！")
                else:
                    print(f"失败 (状态码: {response.status_code})")
            except Exception as e:
                print(f"出错: {e}")

            # 每次下载后休眠，保护服务器，避免被封 IP
            time.sleep(5)

            current_start = current_end

    print("\n" + "!" * 50)
    print("所有 HIGHWAY_ID 的下载任务全部完成！")
    print("!" * 50)



if __name__ == '__main__':
    # Read Data
    data_down()