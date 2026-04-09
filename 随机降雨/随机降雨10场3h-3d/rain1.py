import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. 基础参数配置 ======================
# 沙坪坝区暴雨强度参数
A1 = 1132 / 166.67
C = 0.958
B = 5.408
n = 0.595
r_chicago = 0.4
T_threshold = 24  # 24h分界

# 随机参数：10场降雨，3h~2天
np.random.seed(42)
P_list = np.random.randint(1, 501, size=10)
T_list = np.random.randint(3, 73, size=10)  # 3~72小时

# 固定：每10分钟一个数据点
time_step = 10  # min


# ====================== 2. 雨型生成函数 ======================
def chicago_rainfall(P, T_hour, r, A1, C, B, n):
    T_min = int(T_hour * 60)
    total_steps = int(T_min / time_step)
    ta_steps = int(r * total_steps)

    # 总雨量
    q_avg = 1132 * (1 + C * np.log10(P)) / (T_min + B) ** n
    i_avg = q_avg / 166.67
    H_total = i_avg * T_min

    # 逐10分钟计算雨强 (mm/min)
    i_series = np.zeros(total_steps)
    for step in range(total_steps):
        current_t = step * time_step
        if step < ta_steps:
            dt = ta_steps * time_step - current_t
        else:
            dt = current_t - ta_steps * time_step
        i_series[step] = A1 * (1 + C * np.log10(P)) / (dt + B) ** n

    # 归一化 + 转为 mm/10min
    sum_raw = np.sum(i_series) * time_step
    i_mm_per10min = i_series * (H_total / sum_raw) * time_step

    # 时间序列：0,10,20,30... 分钟
    time_min_series = np.arange(0, total_steps * time_step, time_step)
    return time_min_series, i_mm_per10min, H_total


def multi_peak_rainfall(P, T_hour, A1, C, B, n):
    T_min = int(T_hour * 60)
    total_steps = int(T_min / time_step)

    # 总雨量
    q_avg = 1132 * (1 + C * np.log10(P)) / (T_min + B) ** n
    i_avg = q_avg / 166.67
    H_total = i_avg * T_min

    # 多峰高斯
    num_peaks = np.random.randint(2, 5)
    time_min_series = np.arange(0, total_steps * time_step, time_step)
    i_series = np.zeros(total_steps)

    peak_positions = np.linspace(int(total_steps * 0.1), int(total_steps * 0.9), num_peaks, dtype=int)
    peak_weights = np.array([1.0, 0.7, 0.5, 0.3][:num_peaks])
    peak_weights /= peak_weights.sum()

    for pos, w in zip(peak_positions, peak_weights):
        peak_t = pos * time_step
        width = np.random.randint(int(total_steps * 0.05), int(total_steps * 0.1)) * time_step
        sigma = width / 6
        gauss = np.exp(-((time_min_series - peak_t) ** 2) / (2 * sigma ** 2))
        i_series += gauss * w

    # 背景雨强
    base = np.random.uniform(0.05, 0.15) * i_series.max()
    i_series += base

    # 归一化 + 转为 mm/10min
    sum_raw = np.sum(i_series) * time_step
    i_mm_per10min = i_series * (H_total / sum_raw) * time_step

    return time_min_series, i_mm_per10min, H_total


# ====================== 3. 批量生成 ======================
rain_events = []
for idx in range(10):
    P = P_list[idx]
    T_hour = T_list[idx]

    if T_hour <= T_threshold:
        t_10min, rain_10min, H = chicago_rainfall(P, T_hour, r_chicago, A1, C, B, n)
        rain_type = "芝加哥单峰雨型"
    else:
        t_10min, rain_10min, H = multi_peak_rainfall(P, T_hour, A1, C, B, n)
        rain_type = "多峰雨型(长历时)"

    rain_events.append({
        "index": idx + 1,
        "P": P,
        "T_hour": T_hour,
        "total_rain": round(H, 2),
        "time_10min": t_10min,
        "rain_mm_per10min": rain_10min,
        "rain_type": rain_type
    })

# ====================== 4. 输出与绘图 ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

for event in rain_events:
    idx = event["index"]
    P = event["P"]
    T_hour = event["T_hour"]
    H = event["total_rain"]
    t_10min = event["time_10min"]
    rain_10min = event["rain_mm_per10min"]
    rain_type = event["rain_type"]

    # 打印
    print(f"第{idx}场 | 重现期{P}a | 时长{T_hour}h | 总雨量{H:.2f}mm | {rain_type}")

    # 保存 CSV（Excel 完美打开）
    np.savetxt(
        f"降雨数据_第{idx}场_P{P}a.csv",
        np.column_stack([t_10min, rain_10min]),
        delimiter=",", fmt="%.6f",
        header="时间(min),降雨量(mm/10min)",
        comments=""
    )

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.plot(t_10min, rain_10min, linewidth=1.5, color="#0066cc")
    plt.fill_between(t_10min, rain_10min, alpha=0.3, color="#0066cc")
    plt.title(f"第{idx}场降雨 | 重现期{P}年 | 时长{T_hour}h | 总雨量{H}mm", fontsize=12)
    plt.xlabel("时间 (10min)")
    plt.ylabel("降雨量 (mm/10min)")
    plt.grid(alpha=0.3, linestyle="--")

    # X轴优化
    if T_hour > 24:
        plt.xticks(np.arange(0, max(t_10min)+1, 360))  # 6小时间隔
    else:
        plt.xticks(np.arange(0, max(t_10min)+1, 60))   # 1小时间隔

    plt.tight_layout()
    plt.savefig(f"降雨过程线_第{idx}场_P{P}a.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\n✅ 10场降雨生成完成！")
print("📌 CSV 格式说明：")
print("   第一列：时间(分钟) → 0,10,20,30...")
print("   第二列：降雨量 → mm/10min（每10分钟的降雨量）")
print("   直接用 Excel 打开即可使用！")