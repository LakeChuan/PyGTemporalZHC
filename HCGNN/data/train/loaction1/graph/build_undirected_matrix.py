import numpy as np

def build_adjacency_matrix(txt_path):
    """
    从 undirected_d8_weight1.txt 构建邻接矩阵
    返回：邻接矩阵 (np.array)
    """
    edges = []
    nodes = set()

    # 读取边文件
    with open(txt_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, w = line.split()
            u = int(u)
            v = int(v)
            w = int(w)
            edges.append((u, v, w))
            nodes.add(u)
            nodes.add(v)

    if not nodes:
        raise ValueError("没有有效节点！")

    # 节点总数（最大编号 +1，因为从0开始）
    max_node = max(nodes)
    n = max_node + 1
    print(f"📌 节点总数：{n}")
    print(f"📌 边总数：{len(edges)}")

    # 初始化邻接矩阵
    adj = np.zeros((n, n), dtype=np.int32)

    # 填充矩阵
    for u, v, w in edges:
        adj[u, v] = w
        adj[v, u] = w  # 无向图双向赋值

    return adj

def save_adjacency_matrix(adj, npy_path="undirected_adj_matrix.npy", txt_path="undirected_adj_matrix.txt"):
    """保存邻接矩阵为 npy 和 txt"""
    # 保存 npy（numpy专用，快速读取）
    np.save(npy_path, adj)
    print(f"✅ 邻接矩阵已保存：{npy_path}")

    # 保存 txt（可读格式）
    np.savetxt(txt_path, adj, fmt="%d", delimiter=" ")
    print(f"✅ 邻接矩阵已保存：{txt_path}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 输入你生成的边文件
    txt_file = "undirected_d8_weight1.txt"

    # 构建矩阵
    adj_matrix = build_adjacency_matrix(txt_file)

    # 保存
    save_adjacency_matrix(adj_matrix)

    # 查看矩阵大小
    print(f"\n📊 邻接矩阵形状：{adj_matrix.shape}")