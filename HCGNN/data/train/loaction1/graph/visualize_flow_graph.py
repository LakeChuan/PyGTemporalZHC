import pandas as pd
from pyvis.network import Network

# 读取你的水流有向图数据
df = pd.read_csv(
    "improved_d8_graph.txt",
    sep=r"\s+",  # 加 r 修复转义警告
    header=0
)

# 创建有向图
G = Network(
    height="1000px",
    width="100%",
    directed=True,
    bgcolor="#1a1a1a",
    font_color="white"
)

# 万级节点专用布局
G.force_atlas_2based(
    gravity=-80,
    central_gravity=0.005,
    spring_length=120,
    overlap=0
)

# ===================== 修复核心：先批量添加所有节点 =====================
all_nodes = set(df["from_node"].unique()).union(set(df["to_node"].unique()))
for node in all_nodes:
    G.add_node(str(int(node)), size=8, color="#2196F3")

# ===================== 再添加边 =====================
for _, row in df.iterrows():
    u = str(int(row["from_node"]))
    v = str(int(row["to_node"]))
    G.add_edge(u, v, color="#4CAF50", arrows="to")

# 生成 HTML
G.show("d8_flow_graph.html", notebook=False)
print("✅ 可视化完成！打开 d8_flow_graph.html 即可查看")