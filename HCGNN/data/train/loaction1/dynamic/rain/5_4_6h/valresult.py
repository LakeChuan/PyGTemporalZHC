import numpy as np

# 加载你生成的文件
data = np.load("5_4_6h_feature.npy")
print("shape =", data.shape)

# 取几个关键位置验证
# 任意一个栅格，比如 (0,0)
rain_00 = data[0, 0, 0, :]  # 第0行0列，特征0=降雨，所有时间步
depth_00 = data[0, 0, 1, :] # 第0行0列，特征1=水深，所有时间步

# 再取另一个栅格，比如 (10,10)
rain_1010 = data[10, 10, 0, :]

print("\n===== 验证1：降雨是否全场相同 =====")
print("栅格(0,0)前5个时间步降雨：", rain_00[:5])
print("栅格(10,10)前5个时间步降雨：", rain_1010[:5])

print("\n===== 验证2：水深每个栅格不同 =====")
print("栅格(0,0)前5个时间步水深：", depth_00[:5])

# 检查两个栅格降雨是否完全一样
print("\n===== 验证3：所有栅格降雨是否广播正确 =====")
diff = np.sum(np.abs(rain_00 - rain_1010))
print("两个栅格降雨差值总和：", diff)
if diff < 1e-6:
    print("✅ 降雨全场均匀广播正确！")
else:
    print("❌ 出错，降雨不是全场一致")