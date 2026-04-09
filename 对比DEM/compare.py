import numpy as np
import pandas as pd

# ---------------------- 1. 配置文件路径 ----------------------
txt_path = "nk1kmdem.asc"  # 替换为你的80x80 txt/asc文件路径
excel_path = "南开1km属性.xlsx"  # 替换为你的Excel文件路径
output_path = "diff_results.xlsx"  # 差异结果输出路径

# ---------------------- 2. 读取80x80网格数据 ----------------------
# 跳过前6行头部信息，读取核心数据（假设数据为80行80列）
grid_data = np.loadtxt(txt_path, skiprows=6)  # shape: (80, 80)
# 验证数据维度（防止文件格式错误）
assert grid_data.shape == (80, 80), "网格数据不是80x80格式，请检查文件！"

# ---------------------- 3. 读取Excel数据 ----------------------
# 读取Excel，需包含 row_index、col_index、dem1 三列
df = pd.read_excel(excel_path)
# 校验Excel必要列是否存在
required_cols = ["row_index", "col_index", "dem1"]
assert all(col in df.columns for col in required_cols), "Excel缺少必要列！"

# ---------------------- 4. 逐行比对并收集差异 ----------------------
diff_records = []  # 存储差异数据：[row_idx, col_idx, txt_value, excel_value]

for _, row in df.iterrows():
    # 获取Excel中的行列索引和值（需确保索引在0-79范围内）
    r = int(row["row_index"])
    c = int(row["col_index"])
    excel_val = row["dem1"]

    # 边界校验：防止索引越界
    if 0 <= r < 80 and 0 <= c < 80:
        txt_val = grid_data[r, c]  # 从网格数据中取对应位置值
        # 比对值（考虑浮点数精度差异，允许微小误差）
        if not np.isclose(txt_val, excel_val, atol=1e-6):
            diff_records.append({
                "行索引(row)": r,
                "列索引(col)": c,
                "TXT网格值": txt_val,
                "Excel dem1值": excel_val
            })

# ---------------------- 5. 输出结果 ----------------------
if diff_records:
    # 转换为DataFrame方便查看和保存
    diff_df = pd.DataFrame(diff_records)
    print(f"共发现 {len(diff_records)} 处数据不一致！")
    print(diff_df)
    # 保存差异结果到Excel
    diff_df.to_excel(output_path, index=False)
    print(f"差异结果已保存至：{output_path}")
else:
    print("✅ 所有数据比对一致，无差异！")