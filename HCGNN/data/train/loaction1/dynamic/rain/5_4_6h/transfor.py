import os
import numpy as np


def load_rain_sequence(rain_file_path, total_time_steps):
    """
    加载.rain文件中的降雨序列
    自动截断/补齐到总时间步数（以wd文件数为准）
    """
    with open(rain_file_path, 'r') as f:
        lines = f.readlines()

    # 读取降雨强度（不管文件里写的N，以实际wd数量为准）
    rain_intensity = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 1:
            rain_intensity.append(float(parts[0]))

    rain_intensity = np.array(rain_intensity, dtype=np.float32)

    # 关键：自动匹配到总时间步数
    if len(rain_intensity) > total_time_steps:
        rain_intensity = rain_intensity[:total_time_steps]
    elif len(rain_intensity) < total_time_steps:
        pad = np.zeros(total_time_steps - len(rain_intensity), dtype=np.float32)
        rain_intensity = np.concatenate([rain_intensity, pad])

    return rain_intensity


def load_wd_depth(wd_folder, grid_size=80):
    """
    自动识别所有.wd文件，返回总时间步数 + 深度数据
    返回:
        depth_data: shape (grid_size, grid_size, N)
        total_time_steps: 总时间步数（自动识别）
    """
    # 读取并排序所有wd文件（自动识别数量）
    wd_files = sorted([f for f in os.listdir(wd_folder) if f.endswith('.wd')])
    total_time_steps = len(wd_files)
    print(f"✅ 自动识别 .wd 文件总数 = {total_time_steps}")

    depth_data = np.zeros((grid_size, grid_size, total_time_steps), dtype=np.float32)

    for t, wd_file in enumerate(wd_files):
        wd_path = os.path.join(wd_folder, wd_file)
        with open(wd_path, 'r') as f:
            lines = f.readlines()

        # 第7行开始读取80*80数据
        data_lines = lines[6: 6 + grid_size]
        for row_idx, line in enumerate(data_lines):
            parts = list(map(float, line.strip().split()))
            depth_data[row_idx, :, t] = parts[:grid_size]

    return depth_data, total_time_steps


def main():
    # ====================== 路径配置（不用改，自动识别） ======================
    base_dir = r"E:\Coding\Pytorch\PyGTemporalZHC\HCGNN\data\train\loaction1\dynamic\rain\5_4_6h"
    rain_file_name = "5_4_6h.rain"
    wd_folder_name = "result"
    output_npy_name = "80x80x2_feature.npy"
    grid_size = 80
    # =========================================================================

    # 1. 加载积水深度 + 自动获取总时间步
    wd_folder_path = os.path.join(base_dir, wd_folder_name)
    depth_data, N = load_wd_depth(wd_folder_path, grid_size)

    # 2. 加载并对齐降雨序列
    rain_file_path = os.path.join(base_dir, rain_file_name)
    rain_intensity = load_rain_sequence(rain_file_path, N)
    print(f"✅ 降雨序列已对齐到总时间步 N = {N}")

    # 3. 广播降雨到 80x80
    rain_feature = np.broadcast_to(rain_intensity, (grid_size, grid_size, N))

    # 4. 拼接特征：80x80x2xN
    combined = np.stack([rain_feature, depth_data], axis=2)
    print(f"✅ 最终数据 shape = {combined.shape} (80x80x2xN)")

    # 5. 保存
    output_path = os.path.join(base_dir, output_npy_name)
    np.save(output_path, combined)
    print(f"\n🎉 处理完成！文件已保存到：\n{output_path}")


if __name__ == "__main__":
    main()