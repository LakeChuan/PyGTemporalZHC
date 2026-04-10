import numpy as np
import pandas as pd

# ===================== 1. 读取边数据 =====================
df = pd.read_csv('directed_improved_d8_graph.txt', sep=r'\s+', skiprows=1,
                 names=['from_node', 'to_node', 'slope_weight'])

print("数据读取成功：")
print(f"总边数：{len(df)}")

# ===================== 2. 构建 6400x6400 邻接矩阵 =====================
num_nodes = 6400
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for _, row in df.iterrows():
    u = int(row['from_node'])
    v = int(row['to_node'])
    weight = row['slope_weight']
    adj_matrix[u, v] = weight

print(f"\n邻接矩阵尺寸: {adj_matrix.shape}")
print(f"非零元素数量: {np.count_nonzero(adj_matrix)}")

# ===================== 3. 原本的 npy 保存（不变） =====================
np.save('directed_adjacency_matrix_6400.npy', adj_matrix)
print("\n✅ 邻接矩阵已保存为：directed_adjacency_matrix_6400.npy")

# ===================== 4. 新增：保存为 TXT 文件 =====================
np.savetxt(
    'directed_adjacency_matrix_6400.txt',
    adj_matrix,
    fmt='%.6f',
    delimiter=' ',
    encoding='utf-8'
)
print("✅ 邻接矩阵已保存为：directed_adjacency_matrix_6400.txt")