import os
import numpy as np


def load_rain_sequence(rain_file_path, total_time_steps):
    """加载.rain文件，自动对齐到指定时间步"""
    with open(rain_file_path, 'r') as f:
        lines = f.readlines()

    rain_intensity = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 1:
            rain_intensity.append(float(parts[0]))

    rain_intensity = np.array(rain_intensity, dtype=np.float32)

    # 自动对齐时间步：多截断、少补0
    if len(rain_intensity) > total_time_steps:
        rain_intensity = rain_intensity[:total_time_steps]
    elif len(rain_intensity) < total_time_steps:
        pad = np.zeros(total_time_steps - len(rain_intensity), dtype=np.float32)
        rain_intensity = np.concatenate([rain_intensity, pad])

    return rain_intensity


def load_wd_depth(wd_folder, grid_size=80):
    """加载result文件夹下所有.wd文件，返回深度数据和时间步"""
    wd_files = sorted([f for f in os.listdir(wd_folder) if f.endswith('.wd')])
    total_time_steps = len(wd_files)

    depth_data = np.zeros((grid_size, grid_size, total_time_steps), dtype=np.float32)

    for t, wd_file in enumerate(wd_files):
        wd_path = os.path.join(wd_folder, wd_file)
        with open(wd_path, 'r') as f:
            lines = f.readlines()

        data_lines = lines[6: 6 + grid_size]
        for row_idx, line in enumerate(data_lines):
            parts = list(map(float, line.strip().split()))
            depth_data[row_idx, :, t] = parts[:grid_size]

    return depth_data, total_time_steps


def process_single_rain_folder(folder_path, grid_size=80):
    """处理单个降雨文件夹，返回是否成功"""
    folder_name = os.path.basename(folder_path)
    print(f"\n🔄 正在处理：{folder_name}")

    # 1. 检查文件夹结构是否正确
    # 找.rain文件（和文件夹同名的.rain文件）
    rain_file = None
    for f in os.listdir(folder_path):
        if f.endswith('.rain') and f.startswith(folder_name):
            rain_file = f
            break
    if not rain_file:
        print(f"❌ {folder_name} 未找到对应.rain文件，跳过")
        return False

    # 检查result文件夹
    result_folder = os.path.join(folder_path, 'result')
    if not os.path.isdir(result_folder):
        print(f"❌ {folder_name} 未找到result文件夹，跳过")
        return False

    # 2. 加载水深数据
    try:
        depth_data, N = load_wd_depth(result_folder, grid_size)
    except Exception as e:
        print(f"❌ {folder_name} 加载.wd文件失败：{str(e)}，跳过")
        return False

    # 3. 加载并对齐降雨序列
    rain_file_path = os.path.join(folder_path, rain_file)
    try:
        rain_intensity = load_rain_sequence(rain_file_path, N)
    except Exception as e:
        print(f"❌ {folder_name} 加载.rain文件失败：{str(e)}，跳过")
        return False

    # 4. 广播降雨 + 拼接特征
    rain_feature = np.broadcast_to(rain_intensity, (grid_size, grid_size, N))
    combined_feature = np.stack([rain_feature, depth_data], axis=2)

    # 5. 保存为.npy
    output_file = f"{folder_name}_feature.npy"
    output_path = os.path.join(folder_path, output_file)
    np.save(output_path, combined_feature)

    print(f"✅ {folder_name} 处理完成！shape={combined_feature.shape}，已保存到：{output_path}")
    return True


def main():
    # ====================== 根目录配置（直接用你当前的路径） ======================
    root_rain_dir = r"E:\Coding\Pytorch\PyGTemporalZHC\HCGNN\data\train\loaction1\dynamic\rain"
    grid_size = 80
    # =========================================================================

    # 1. 遍历所有子文件夹（降雨场次）
    all_folders = [
        os.path.join(root_rain_dir, d)
        for d in os.listdir(root_rain_dir)
        if os.path.isdir(os.path.join(root_rain_dir, d))
    ]

    print(f"📦 共找到 {len(all_folders)} 个降雨文件夹，开始批量处理...")
    success_count = 0
    fail_count = 0

    # 2. 逐个处理
    for folder in all_folders:
        if process_single_rain_folder(folder, grid_size):
            success_count += 1
        else:
            fail_count += 1

    # 3. 最终统计
    print(f"\n🎉 批量处理完成！成功：{success_count} 场，失败：{fail_count} 场")


if __name__ == "__main__":
    main()