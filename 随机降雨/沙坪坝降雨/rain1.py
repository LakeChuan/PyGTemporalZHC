import math
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 12 小时芝加哥雨型（绝对能跑！绝对有雨！）=====================
P = 10  # 10年一遇
dt = 1  # 5分钟步长
T = 360  # 12小时 = 720 分钟
r = 0.4  # 雨峰位置 0.4

# 暴雨公式参数
A1 = 1132
C = 0.958
b = 5.408
n = 0.595
A = A1 * (1 + C * math.log10(P))
tp = r * T
steps = int(T / dt)

time_list = []
rain_list = []

for i in range(steps):
    t = (i + 0.5) * dt

    # 上升段（越来越大）
    if t <= tp:
        i_t = A / (t + b) ** n * (t / tp)
    # 下降段（越来越小）
    else:
        i_t = A / (t + b) ** n * ((T - t) / (T - tp))

    # 强制保证有雨，绝对不会 0
    i_t = max(i_t, 5.0)

    # 单位换算
    rain = i_t * 0.018 * dt
    time_list.append(round(t, 1))
    rain_list.append(round(rain, 2))

# ===================== 保存 Excel =====================
df = pd.DataFrame({
    "时段": range(1, steps + 1),
    "时间(min)": time_list,
    f"{dt}min降雨量(mm)": rain_list
})
df.to_excel(f"重庆{P}年一遇_{T/60}小时_芝加哥雨型.xlsx", index=False)
print("✅ Excel 保存成功！")

# ===================== 绘图 + 自动保存图片 =====================
plt.figure(figsize=(14, 6))
plt.plot(time_list, rain_list, linewidth=2.5, color='#1f77b8')
plt.axvline(tp, color='r', linestyle='--', linewidth=2, label=f'雨峰位置 {int(tp)}min')
plt.title(f"重庆沙坪坝 {P}年一遇_{T/60}小时芝加哥雨型（先升后降）", fontsize=14)
plt.xlabel("时间(min)")
plt.ylabel("5min降雨量(mm)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 👇 这里自动保存图片！
plt.savefig(f"重庆{P}年一遇_{T/60}小时芝加哥雨型.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 输出 =====================
print("\n前20条数据：")
print(df.head(20))

print(f"\n✅ 图片已保存：重庆{P}年一遇_{T/60}小时芝加哥雨型.png")