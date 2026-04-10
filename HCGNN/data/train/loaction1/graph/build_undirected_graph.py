import numpy as np
from collections import defaultdict


def load_dem_asc(file_path):
    """加载ASCII DEM栅格数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1])

        rows = int(header['nrows'])
        cols = int(header['ncols'])
        nodata = header['nodata_value']
        cellsize = header['cellsize']

        dem = []
        for _ in range(rows):
            row = list(map(float, f.readline().strip().split()))
            dem.append(row)

        dem_array = np.array(dem, dtype=np.float32)
        dem_array[dem_array == nodata] = np.nan

        return dem_array, cellsize, rows, cols


def coord2node(i, j, cols):
    """先行后列：栅格坐标 → 一维节点编号"""
    return i * cols + j


def d8_undirected_uniform_graph(dem, rows, cols):
    """
    ✅ 新版 D8 算法（按你的最新要求）
    规则：
    1. 中心栅格 和 8邻域栅格 都是有效值
    2. 生成一条 无向边（双向都存）
    3. 边的权值 weight = 1
    """
    # 8个方向（只需要偏移，不需要距离）
    neighbors = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    # 有向图存储无向边
    digraph = defaultdict(list)

    # 初始化所有节点
    for i in range(rows):
        for j in range(cols):
            u = coord2node(i, j, cols)
            digraph[u] = []

    # 遍历每个栅格
    for i in range(rows):
        for j in range(cols):
            z_current = dem[i, j]
            u = coord2node(i, j, cols)

            # 当前是无效值 → 跳过
            if np.isnan(z_current):
                continue

            # 遍历8个方向
            for di, dj in neighbors:
                ni, nj = i + di, j + dj

                # 越界跳过
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue

                z_neighbor = dem[ni, nj]
                # 邻域是无效值 → 跳过
                if np.isnan(z_neighbor):
                    continue

                # ===================== 核心规则 =====================
                # 双方都有效 → 生成无向边（双向都存，权重=1）
                v = coord2node(ni, nj, cols)
                weight = 1

                # 无向边 = 双向有向边
                if (v, weight) not in digraph[u]:
                    digraph[u].append((v, weight))
                if (u, weight) not in digraph[v]:
                    digraph[v].append((u, weight))
                # ====================================================

    return digraph


def save_result(digraph, output="undirected_d8_weight1.txt"):
    """保存无向图：起点  终点  权重=1"""
    with open(output, 'w', encoding='utf-8') as f:
        f.write("from_node\tto_node\tweight\n")
        for u in sorted(digraph.keys()):
            for (v, w) in digraph[u]:
                f.write(f"{u}\t{v}\t{w}\n")
    print(f"✅ 无向D8图已保存：{output}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 加载DEM
    dem_path = "nk1kmdem.asc"
    dem, cellsize, rows, cols = load_dem_asc(dem_path)
    print(f"📊 DEM大小：{rows}×{cols}")

    # 数据统计
    total_cells = rows * cols
    invalid_count = np.sum(np.isnan(dem))
    valid_count = total_cells - invalid_count
    print(f"\n====== DEM 数据统计 ======")
    print(f"总栅格数: {total_cells}")
    print(f"有效栅格数: {valid_count}")
    print(f"无效栅格数: {invalid_count}")

    # 2. 运行新版无向D8算法
    digraph = d8_undirected_uniform_graph(dem, rows, cols)

    # 3. 统计结果
    total_nodes = len(digraph)
    total_edges = sum(len(edges) for edges in digraph.values())
    print(f"\n✅ 构建完成：节点数={total_nodes}，总边数={total_edges}")

    # 4. 保存结果
    save_result(digraph)

    # 预览前10条边
    print("\n🔍 前10条边预览：")
    cnt = 0
    for u in sorted(digraph):
        for v, w in digraph[u]:
            print(f"{u:4d} ↔ {v:4d} | 权重={w}")
            cnt += 1
            if cnt >= 10:
                break
        if cnt >= 10:
            break