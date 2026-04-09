import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ====================== 全局参数配置 ======================
A1 = 1132 / 166.67
C = 0.958
B = 5.408
n = 0.595

P_LIST = [5, 10, 20, 50, 100]
R_LIST = [0.4, 0.6]
T_HOUR_LIST = [6, 7, 8]
DT_MIN = 5
DT_S = 300

np.random.seed(42)


# ====================== 雨型生成函数 ======================
def chicago_rain_single(P, T_hour, r, A1, C, B, n):
    T_total_min = T_hour * 60
    total_steps = int(T_total_min / DT_MIN)
    ta_steps = int(r * total_steps)

    q_avg = 1132 * (1 + C * np.log10(P)) / (T_total_min + B) ** n
    i_avg_mmmin = q_avg / 166.67
    H_total = i_avg_mmmin * T_total_min

    i_mmmin_series = np.zeros(total_steps)
    for step in range(total_steps):
        t_now_min = step * DT_MIN
        if step < ta_steps:
            ddt = ta_steps * DT_MIN - t_now_min
        else:
            ddt = t_now_min - ta_steps * DT_MIN
        i_mmmin_series[step] = A1 * (1 + C * np.log10(P)) / (ddt + B) ** n

    rain_per_5min = i_mmmin_series * DT_MIN
    sum_rain = np.sum(rain_per_5min)
    rain_per_5min = rain_per_5min * (H_total / sum_rain)

    intensity_mmh = rain_per_5min * 12
    time_sec_arr = np.arange(DT_S, (total_steps + 1) * DT_S, DT_S)
    rain_mm = rain_per_5min

    return time_sec_arr, intensity_mmh, rain_mm, round(H_total, 2), total_steps


# ====================== 保存文件函数 ======================
def save_rainfall_data(fold_name, i_mmh, t_s, rain_mm, total_steps):
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)

    data = np.column_stack([i_mmh, t_s, rain_mm])
    df = pd.DataFrame(data, columns=["降雨强度(mm/h)", "累计时间(s)", "时段降雨量(mm)"])
    df = df.round(3)

    # 1. 保存 Excel
    excel_path = os.path.join(fold_name, f"{fold_name}.xlsx")
    df.to_excel(excel_path, index=False)

    # 2. 保存 TXT（第一列顶格）
    txt_path = os.path.join(fold_name, f"{fold_name}.txt")
    with open(txt_path, 'w') as f:
        for row in data:
            line = f"{row[0]:.3f} {int(row[1]):10d} {row[2]:.3f}\n"
            f.write(line)

    # 3. 保存 .rain 文件（注释行 + 1个空格间隔）
    rain_path = os.path.join(fold_name, f"{fold_name}.rain")
    with open(rain_path, 'w') as f:
        f.write(f"# {total_steps * 300}\n")  # 新增注释行
        f.write(f"{total_steps} seconds\n")  # 仅1个空格
        for intensity, time_sec in zip(i_mmh, t_s):
            line = f"{intensity:.3f} {int(time_sec):10d}\n"
            f.write(line)


# ====================== 绘图函数 ======================
def plot_rainfall(fold_name, t_s, i_mmh, H_total):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 5))
    plt.plot(t_s / 3600, i_mmh, color="#0066cc", lw=1.5)
    plt.fill_between(t_s / 3600, i_mmh, alpha=0.3, color="#0066cc")
    plt.title(f"{fold_name} | 总雨量{H_total}mm", fontsize=12)
    plt.xlabel("时间(h)")
    plt.ylabel("降雨强度(mm/h)")
    plt.grid(alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_name, f"{fold_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ====================== 主执行函数 ======================
def main():
    # 用于保存所有生成的文件夹名称
    name_list = []

    for P in P_LIST:
        for r in R_LIST:
            for T in T_HOUR_LIST:
                t_s, i_mmh, rain_mm, H_total, total_steps = chicago_rain_single(P, T, r, A1, C, B, n)
                # 命名去掉小数：0.4→4  0.5→5  0.6→6
                r_int = int(round(r * 10))
                fold_name = f"{P}_{r_int}_{T}h"

                # 把名称加入列表
                name_list.append(fold_name)

                save_rainfall_data(fold_name, i_mmh, t_s, rain_mm, total_steps)
                plot_rainfall(fold_name, t_s, i_mmh, H_total)
                print(f"✅ 已生成：{fold_name}")

    # ====================== 保存所有名称到 txt ======================
    with open("降雨文件名列表.txt", "w", encoding="utf-8") as f:
        for name in name_list:
            f.write(name + "\n")

    print("\n🎉 所有降雨生成完成！")
    print("📄 已生成：降雨文件名列表.txt")
    print("📌 最终格式规范：")
    print("  命名无小数：5_4_6h")
    print("  .rain 第1行：#注释行")
    print("  .rain 第2行：总行数 单位（1个空格）")
    print("  .txt/.rain 第1列顶格无空格")


# ====================== 程序入口 ======================
if __name__ == '__main__':
    main()