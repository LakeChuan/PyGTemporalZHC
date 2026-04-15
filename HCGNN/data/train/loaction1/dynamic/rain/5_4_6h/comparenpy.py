import numpy as np


def compare_npy_files(file1: str, file2: str, output_txt: str = "diff_result.txt") -> None:
    """
    对比两个npy文件的数据是否相等，不相等则记录差异并保存到txt

    Args:
        file1: 第一个npy文件路径
        file2: 第二个npy文件路径
        output_txt: 差异结果保存的txt文件路径
    """
    # 1. 加载npy文件
    try:
        data1 = np.load(file1)
        data2 = np.load(file2)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    # 2. 先对比形状是否一致
    if data1.shape != data2.shape:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(f"两个文件形状不一致！\n")
            f.write(f"文件1 {file1} 形状: {data1.shape}\n")
            f.write(f"文件2 {file2} 形状: {data2.shape}\n")
        print(f"形状不一致，结果已保存到 {output_txt}")
        return

    # 3. 逐元素对比，得到差异掩码
    diff_mask = ~np.isclose(data1, data2)  # 用isclose处理浮点误差，避免精度问题
    diff_count = np.sum(diff_mask)

    # 4. 无差异的情况
    if diff_count == 0:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(f"两个文件数据完全相等！\n")
            f.write(f"文件1: {file1}\n")
            f.write(f"文件2: {file2}\n")
            f.write(f"数据形状: {data1.shape}\n")
        print(f"数据完全相等，结果已保存到 {output_txt}")
        return

    # 5. 有差异的情况，提取所有不相等的位置和数值
    diff_indices = np.where(diff_mask)  # 获取所有不相等元素的索引
    diff_values1 = data1[diff_mask]
    diff_values2 = data2[diff_mask]
    diff_abs = np.abs(diff_values1 - diff_values2)

    # 6. 写入txt文件
    with open(output_txt, "w", encoding="utf-8") as f:
        # 写入基本信息
        f.write("=" * 60 + "\n")
        f.write("npy文件对比结果\n")
        f.write(f"文件1: {file1}\n")
        f.write(f"文件2: {file2}\n")
        f.write(f"数据形状: {data1.shape}\n")
        f.write(f"不相等元素总数: {diff_count}\n")
        f.write(f"最大绝对差异: {np.max(diff_abs):.10f}\n")
        f.write(f"平均绝对差异: {np.mean(diff_abs):.10f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("不相等元素详情（索引, 文件1数值, 文件2数值, 绝对差异）:\n")
        f.write("-" * 80 + "\n")

        # 写入每个不相等元素的详情（最多显示前1000条，避免文件过大）
        max_show = 1000
        for i in range(min(diff_count, max_show)):
            # 处理多维索引，转为元组格式
            idx = tuple(dim[i] for dim in diff_indices)
            val1 = diff_values1[i]
            val2 = diff_values2[i]
            abs_diff = diff_abs[i]
            f.write(f"索引 {idx}: 文件1={val1:.10f}, 文件2={val2:.10f}, 差异={abs_diff:.10f}\n")

        # 如果差异数量超过1000，提示截断
        if diff_count > max_show:
            f.write(f"\n（仅显示前{max_show}条差异，总差异数为{diff_count}）\n")

    print(f"对比完成！发现 {diff_count} 个不相等元素，结果已保存到 {output_txt}")


# ------------------- 主程序调用 -------------------
if __name__ == "__main__":
    # 替换为你的两个npy文件路径
    file_path1 = "5_4_6h_feature.npy"
    file_path2 = "THWC_feature.npy"
    # 结果保存的txt路径
    output_path = "npy_diff_result.txt"

    # 执行对比
    compare_npy_files(file_path1, file_path2, output_path)