"""
MTGNN 完整运行示例（不依赖内置数据集，手动构建时空图）
适配所有版本的 torch_geometric_temporal（只要包含 MTGNN 类）
"""
import torch
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import MTGNN


# ===================== 1. 手动构建模拟时空图数据 =====================
def build_synthetic_temporal_graph(num_nodes=100, num_timesteps=1000, in_channels=2):
    """
    构建模拟的静态拓扑时空图数据（替代内置数据集）
    :param num_nodes: 节点数
    :param num_timesteps: 时间步数
    :param in_channels: 节点特征维度（多变量数）
    :return: StaticGraphTemporalSignal（和内置数据集格式一致）
    """
    # 1. 构建静态图拓扑（随机生成边索引，模拟路网）
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))  # 每个节点平均4条边
    # 2. 生成时间序列节点特征（随机值，模拟交通流量）
    features = [torch.randn(num_nodes, in_channels) for _ in range(num_timesteps)]
    # 3. 生成时间序列标签（预测下一时间步，简单设为特征的偏移）
    targets = [features[t+1] if t < num_timesteps-1 else features[t] for t in range(num_timesteps)]

    # 封装为静态拓扑时空图信号（核心格式）
    temporal_signal = StaticGraphTemporalSignal(
        edge_index=edge_index.numpy(),
        edge_weight=None,
        features=[f.numpy() for f in features],
        targets=[t.numpy() for t in targets]
    )
    return temporal_signal


def data_loader(dataset, batch_size=32, shuffle=True):
    """批量数据加载器（和之前一致）"""
    snapshots = list(dataset)
    if shuffle:
        np.random.shuffle(snapshots)
    for i in range(0, len(snapshots), batch_size):
        batch_snapshots = snapshots[i:i+batch_size]
        x = torch.stack([s.x for s in batch_snapshots], dim=0).float()
        edge_index = batch_snapshots[0].edge_index
        y = torch.stack([s.y for s in batch_snapshots], dim=0).float()
        yield x, edge_index, y


class MTGNNModel(torch.nn.Module):
    """MTGNN模型（和之前一致）"""
    def __init__(self, num_nodes, in_channels, out_channels):
        super().__init__()
        self.mtgnn = MTGNN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=64,
            num_layers=2,
            kernel_size=3,
            k=3,
            dropout=0.1,
            activation="relu"
        )

    def forward(self, x, edge_index):
        return self.mtgnn(x, edge_index)


def train_epoch(model, loader, optimizer, criterion, device):
    """训练函数（和之前一致）"""
    model.train()
    total_loss = 0.0
    for x, edge_index, y in loader:
        x = x.to(device)
        edge_index = edge_index.to(device)
        y = y.to(device)

        pred = model(x, edge_index)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def test_epoch(model, loader, criterion, device):
    """测试函数（和之前一致）"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, edge_index, y in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            y = y.to(device)

            pred = model(x, edge_index)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    # 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    epochs = 5
    batch_size = 32

    # 手动构建模拟数据（替代内置数据集）
    num_nodes = 100
    in_channels = 2
    out_channels = 2  # 输出维度和输入一致（模拟预测）
    dataset = build_synthetic_temporal_graph(num_nodes=num_nodes, in_channels=in_channels)

    # 划分训练/测试集
    train_size = int(0.8 * len(dataset))
    train_dataset = StaticGraphTemporalSignal(
        edge_index=dataset.edge_index,
        edge_weight=dataset.edge_weight,
        features=dataset.features[:train_size],
        targets=dataset.targets[:train_size]
    )
    test_dataset = StaticGraphTemporalSignal(
        edge_index=dataset.edge_index,
        edge_weight=dataset.edge_weight,
        features=dataset.features[train_size:],
        targets=dataset.targets[train_size:]
    )

    # 初始化模型/优化器/损失函数
    model = MTGNNModel(num_nodes, in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 训练+测试
    print("\n开始训练...")
    for epoch in range(1, epochs + 1):
        train_loader = data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = data_loader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = test_epoch(model, test_loader, criterion, device)

        print(f"Epoch {epoch:02d} | 训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f}")

    # 单样本预测验证
    print("\n单样本预测验证...")
    model.eval()
    test_snapshot = next(iter(test_dataset))
    x = test_snapshot.x.unsqueeze(0).float().to(device)
    edge_index = test_snapshot.edge_index.to(device)

    with torch.no_grad():
        pred = model(x, edge_index)

    print(f"预测结果shape: {pred.shape} (格式：[batch, 节点数, 输出维度])")
    print(f"真实标签shape: {test_snapshot.y.shape}")
    print(f"\n前5个节点的预测值（第一维特征）: {pred[0, :5, 0].cpu().numpy()}")
    print(f"前5个节点的真实值（第一维特征）: {test_snapshot.y[:5, 0].numpy()}")