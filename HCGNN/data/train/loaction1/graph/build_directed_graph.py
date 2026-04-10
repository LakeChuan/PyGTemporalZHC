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
    """先行后列：栅格坐标 → 一维节点编号 0~6399"""
    return i * cols + j


def improved_d8_graph(dem, cellsize, rows, cols):
    """
    🔥 改进型D8算法（完全按你的规则）
    规则：
    1. 当前高程 ≥ 邻域高程 → 生成边
    2. 当前高程 > 邻域高程 → 正常算坡度
    3. 当前高程 = 邻域高程 → 高差=0.001算坡度
    4. 当前高程 < 邻域高程 → 无边
    """
    # 8个方向：行偏移、列偏移、距离系数(正交=1，对角=√2)
    neighbors = [
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2)),
        (0, 1, 1.0),
        (1, 1, np.sqrt(2)),
        (1, 0, 1.0),
        (1, -1, np.sqrt(2)),
        (0, -1, 1.0),
        (-1, -1, np.sqrt(2))
    ]

    # 有向图：{ 起点ID: [(终点ID, 坡度), ...] }
    digraph = defaultdict(list)

    # ===================== 先初始化所有 0~6399 节点（修复空节点不显示问题） =====================
    for i in range(rows):
        for j in range(cols):
            u = coord2node(i, j, cols)
            digraph[u] = []  # 强制每个编号都存在，空列表也保留
    # ========================================================================================

    for i in range(rows):
        for j in range(cols):
            z_current = dem[i, j]
            u = coord2node(i, j, cols)

            # 无效值直接跳过（但节点已经创建，只是没有边）
            if np.isnan(z_current):
                continue

            # 遍历8个方向
            for di, dj, d_coeff in neighbors:
                ni, nj = i + di, j + dj

                # 邻域越界 → 跳过
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue

                z_neighbor = dem[ni, nj]
                if np.isnan(z_neighbor):
                    continue

                # ===================== 核心规则 =====================
                if z_current >= z_neighbor:
                    # 满足条件：生成边
                    if z_current > z_neighbor:
                        delta = z_current - z_neighbor
                    else:
                        delta = 0.001  # 相等时强制高差

                    distance = cellsize * d_coeff
                    slope = delta / distance
                    v = coord2node(ni, nj, cols)
                    digraph[u].append((v, slope))
                # else: z_current < z_neighbor → 不生成边
                # ====================================================

    return digraph


def save_d8_result(digraph, output="directed_improved_d8_graph.txt"):
    """保存最终有向图：起点  终点  坡度权重"""
    with open(output, 'w', encoding='utf-8') as f:
        f.write("from_node\tto_node\tslope_weight\n")
        for u in sorted(digraph.keys()):
            for (v, w) in digraph[u]:
                f.write(f"{u}\t{v}\t{w:.6f}\n")
    print(f"✅ 改进D8有向图已保存：{output}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 加载DEM
    dem_path = "nk1kmdem.asc"
    dem, cellsize, rows, cols = load_dem_asc(dem_path)
    print(f"📊 DEM大小：{rows}×{cols}")

    # ===================== 新增：验证有效/无效数据 =====================
    total_cells = rows * cols
    nan_mask = np.isnan(dem)
    invalid_count = np.sum(nan_mask)
    valid_count = total_cells - invalid_count

    print(f"\n====== DEM 数据有效性验证 ======")
    print(f"总栅格数: {total_cells}")
    print(f"有效栅格数: {valid_count}")
    print(f"无效(NODATA)栅格数: {invalid_count}")

    # 打印所有无效值坐标
    if invalid_count > 0:
        invalid_positions = np.argwhere(nan_mask)
        print(f"无效栅格坐标 (行, 列):")
        for pos in invalid_positions:
            print(f"  {pos}")
    else:
        print("✅ 无任何无效栅格，全部 6400 个都是有效数据")
    # ==================================================================

    # 2. 🔥 运行改进版D8算法
    digraph = improved_d8_graph(dem, cellsize, rows, cols)

    # 3. 统计结果
    total_nodes = len(digraph)
    total_edges = sum(len(edges) for edges in digraph.values())
    print(f"\n✅ 构建完成：节点数={total_nodes}，总边数={total_edges}")

    # 4. 保存结果
    save_d8_result(digraph)

    # 预览前10条边
    print("\n🔍 前10条边预览：")
    cnt = 0
    for u in sorted(digraph):
        for v, w in digraph[u]:
            print(f"{u:4d} → {v:4d} | 坡度={w:.6f}")
            cnt += 1
            if cnt >= 10:
                break
        if cnt >= 10:
            break