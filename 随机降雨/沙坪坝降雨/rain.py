import math
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 正常显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# ===================== 1. 基础参数设置 =====================
P = 20  # 设计重现期：10年一遇
dt = 5  # 时间步长：5分钟
total_min = 1440  # 总时长：1天 = 1440分钟
n_steps = total_min // dt  # 总时段数：288个

# 重庆主城区暴雨强度公式参数
A1 = 1132
C = 0.958
b = 5.408
n = 0.595

# ===================== 2. 计算暴雨强度与降雨量 =====================
# 计算公式常数项
A = A1 * (1 + C * math.log10(P))

# 逐时段计算
time_list = []  # 时间（分钟）
rain_intensity = []  # 暴雨强度 L/(s·hm²)
rainfall_mm = []  # 5min降雨量 mm

for i in range(1, n_steps + 1):
    # 时段中点历时（更准确）
    t = i * dt - dt / 2
    # 暴雨强度
    q = A / (t + b) ** n
    # 5min降雨量换算：R(mm) = q × 1.8
    r = q * 1.8

    time_list.append(round(t, 1))
    rain_intensity.append(round(q, 2))
    rainfall_mm.append(round(r, 2))

# ===================== 3. 保存为 Excel 文件 =====================
df = pd.DataFrame({
    "时段序号": range(1, n_steps + 1),
    "累计时间(min)": time_list,
    "暴雨强度(L/(s·hm²))": rain_intensity,
    "5min降雨量(mm)": rainfall_mm
})
df.to_excel("重庆20年一遇_5min降雨序列.xlsx", index=False)
print("✅ Excel 文件已保存：重庆20年一遇_5min降雨序列.xlsx")

# ===================== 4. 绘制降雨历时曲线图 =====================
plt.figure(figsize=(14, 6))
plt.plot(time_list, rainfall_mm, color="#1f77b4", linewidth=1.5, marker=".", markersize=2)
plt.title("重庆主城区 20年一遇暴雨强度-降雨历时曲线（5min步长）", fontsize=14)
plt.xlabel("降雨历时（分钟）", fontsize=12)
plt.ylabel("5min时段降雨量（mm）", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("降雨历时曲线20.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 5. 输出前10条数据预览 =====================
print("\n📊 前10条降雨序列预览：")
print(df.head(10))