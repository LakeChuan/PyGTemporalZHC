import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ===================== 全局配置（与论文和你的数据对齐） =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 12  # 输入序列长度：5min×12=1小时
PRED_LEN = 12  # 预测长度：5min×12=1小时
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
HID_CHANNELS = 64
K = 3  # 扩散步数（论文默认）
NUM_LAYERS = 2  # DCGRU层数
NUM_NODES = 207  # METR-LA传感器数量
DATA_ROOT = "./METR-LA"
ADJ_PATH = "./sensor_graph/adj_mx.pkl"


# ===================== 数据集：直接读取你现有的 npz 文件 =====================
class METRLADataset(Dataset):
    def __init__(self, npz_path, scaler=None):
        data = np.load(npz_path)
        self.x = data['x']  # (num_samples, seq_len, num_nodes, 2)
        self.y = data['y']  # (num_samples, pred_len, num_nodes, 2)

        # 只取速度特征（第0维）
        self.x = self.x[..., 0:1]  # (N, T, N_nodes, 1)
        self.y = self.y[..., 0:1]

        # 标准化（用训练集的scaler）
        if scaler is not None:
            N, T, N_nodes, C = self.x.shape
            self.x = scaler.transform(self.x.reshape(-1, C)).reshape(N, T, N_nodes, C)
            N, T, N_nodes, C = self.y.shape
            self.y = scaler.transform(self.y.reshape(-1, C)).reshape(N, T, N_nodes, C)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])


# ===================== 加载邻接矩阵（用你现有的 adj_mx.pkl） =====================
def load_adj(pkl_path):
    with open(pkl_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    adj_mx = adj_mx.astype(np.float32)
    return torch.FloatTensor(adj_mx).to(DEVICE)


# ===================== 扩散卷积 =====================
class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K = K
        self.theta1 = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.theta2 = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.theta1)
        nn.init.xavier_uniform_(self.theta2)

    def forward(self, x, adj):
        # 计算转移矩阵
        out_deg = torch.sum(adj, dim=1, keepdim=True)
        out_deg[out_deg == 0] = 1.0
        trans = adj / out_deg

        in_deg = torch.sum(adj, dim=0, keepdim=True).t()
        in_deg[in_deg == 0] = 1.0
        trans_inv = adj.t() / in_deg

        out = 0
        # 前向扩散
        x_prev = x
        for k in range(self.K):
            out += torch.matmul(x_prev, self.theta1[k])
            x_prev = torch.matmul(trans, x_prev)

        # 反向扩散
        x_prev = x
        for k in range(self.K):
            out += torch.matmul(x_prev, self.theta2[k])
            x_prev = torch.matmul(trans_inv, x_prev)

        return out + self.bias


# ===================== DCGRU单元 =====================
class DCGRUCell(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.gate_conv = DiffusionConv(in_channels + hid_channels, 2 * hid_channels, K)
        self.cand_conv = DiffusionConv(in_channels + hid_channels, hid_channels, K)

    def forward(self, x, h_prev, adj):
        combined = torch.cat([x, h_prev], dim=-1)
        gate = torch.sigmoid(self.gate_conv(combined, adj))
        r, u = torch.chunk(gate, 2, dim=-1)

        combined_c = torch.cat([x, r * h_prev], dim=-1)
        c = torch.tanh(self.cand_conv(combined_c, adj))

        return u * h_prev + (1 - u) * c


# ===================== Encoder =====================
class DCRNNEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.layers = nn.ModuleList([DCGRUCell(in_channels, hid_channels)])
        for _ in range(NUM_LAYERS - 1):
            self.layers.append(DCGRUCell(hid_channels, hid_channels))

    def forward(self, x, adj):
        B, T, N, C = x.shape
        h = [torch.zeros(B, N, HID_CHANNELS).to(DEVICE) for _ in range(NUM_LAYERS)]

        for t in range(T):
            x_t = x[:, t]
            for i, layer in enumerate(self.layers):
                h[i] = layer(x_t, h[i], adj)
                x_t = h[i]
        return h


# ===================== Decoder =====================
class DCRNNDecoder(nn.Module):
    def __init__(self, hid_channels, out_channels):
        super().__init__()
        self.layers = nn.ModuleList([DCGRUCell(out_channels, hid_channels)])
        for _ in range(NUM_LAYERS - 1):
            self.layers.append(DCGRUCell(hid_channels, hid_channels))
        self.proj = DiffusionConv(hid_channels, out_channels, K)

    def forward(self, h_enc, adj, pred_len):
        B, N, _ = h_enc[0].shape
        outputs = []
        x_t = torch.zeros(B, N, 1).to(DEVICE)
        h = h_enc

        for _ in range(pred_len):
            for i, layer in enumerate(self.layers):
                h[i] = layer(x_t, h[i], adj)
                x_t = h[i]
            x_t = self.proj(x_t, adj)
            outputs.append(x_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)


# ===================== DCRNN完整模型 =====================
class DCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DCRNNEncoder(1, HID_CHANNELS)
        self.decoder = DCRNNDecoder(HID_CHANNELS, 1)

    def forward(self, x, adj):
        h_enc = self.encoder(x, adj)
        return self.decoder(h_enc, adj, PRED_LEN)


# ===================== 评估指标（和论文一致的 masked 版本） =====================
def masked_mae(pred, label, mask_val=0.0):
    mask = (label != mask_val).float()
    mask /= mask.mean()
    return (F.l1_loss(pred, label, reduction='none') * mask).mean()


def masked_rmse(pred, label, mask_val=0.0):
    mask = (label != mask_val).float()
    mask /= mask.mean()
    return torch.sqrt((F.mse_loss(pred, label, reduction='none') * mask).mean())


def masked_mape(pred, label, mask_val=0.0):
    mask = (label != mask_val).float()
    mask /= mask.mean()
    return (torch.abs((pred - label) / label) * mask).mean()


# ===================== 训练/测试 =====================
def train_epoch(model, loader, adj, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x, adj)
        loss = masked_mae(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def test_epoch(model, loader, adj):
    model.eval()
    mae, rmse, mape = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x, adj)
            mae += masked_mae(pred, y).item() * x.size(0)
            rmse += masked_rmse(pred, y).item() * x.size(0)
            mape += masked_mape(pred, y).item() * x.size(0)
    cnt = len(loader.dataset)
    return mae / cnt, rmse / cnt, mape / cnt


# ===================== 主函数（直接用你现有的数据文件） =====================
def main():
    # 1. 加载邻接矩阵
    print("Loading adjacency matrix from sensor_graph/adj_mx.pkl...")
    adj = load_adj(ADJ_PATH)

    # 2. 加载训练数据，计算scaler
    print("Loading datasets from METR-LA/*.npz...")
    train_npz = np.load(os.path.join(DATA_ROOT, "train.npz"))
    train_data_all = train_npz['x'][..., 0].reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(train_data_all)

    # 3. 创建Dataset和DataLoader
    train_dataset = METRLADataset(os.path.join(DATA_ROOT, "train.npz"), scaler)
    val_dataset = METRLADataset(os.path.join(DATA_ROOT, "val.npz"), scaler)
    test_dataset = METRLADataset(os.path.join(DATA_ROOT, "test.npz"), scaler)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=True)

    # 4. 模型初始化
    model = DCRNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    # 5. 训练
    print(f"Start training on {DEVICE}...")
    best_mae = float('inf')
    train_losses, val_maes = [], []

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, adj, optimizer)
        val_mae, val_rmse, val_mape = test_epoch(model, val_loader, adj)

        train_losses.append(train_loss)
        val_maes.append(val_mae)

        # 保存最优模型
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), 'DCRNN_METR-LA_best.pth')

        scheduler.step(val_mae)
        print(f"Epoch {epoch + 1:2d} | Time:{time.time() - t0:.1f}s | TrainLoss:{train_loss:.4f}")
        print(f"Val MAE:{val_mae:.4f} RMSE:{val_rmse:.4f} MAPE:{val_mape:.4f}\n")

    # 6. 最终测试
    model.load_state_dict(torch.load('DCRNN_METR-LA_best.pth'))
    test_mae, test_rmse, test_mape = test_epoch(model, test_loader, adj)
    print("=" * 50)
    print("Final Test Results (METR-LA)")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.4f}")
    print("=" * 50)

    # 7. 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_maes, label='Val MAE')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('DCRNN Training Curve')
    plt.legend(), plt.grid(True)
    plt.savefig('DCRNN_loss_curve.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()