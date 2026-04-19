import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pandas as pd

# ======================== 可配置参数：在这里切换数据集 =========================
# 可选值："METR-LA" 或 "PEMS-BAY"
DATASET = "PEMS-BAY"

# ======================== 超参数（根据数据集自动调整） =========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 12
PRED_LEN = 12
INPUT_DIM = 1
OUTPUT_DIM = 1
RNN_UNITS = 32
NUM_LAYERS = 2
MAX_DIFFUSION_STEP = 2
BATCH_SIZE = 32
EPOCHS = 2
LR = 0.001
PATIENCE = 15

# 根据数据集自动设置节点数和邻接矩阵路径
if DATASET == "METR-LA":
    NUM_NODES = 207
    ADJ_PATH = "../DATA/JC/sensor_graph/adj_mx.pkl"
elif DATASET == "PEMS-BAY":
    NUM_NODES = 325
    ADJ_PATH = "../DATA/JC/sensor_graph/adj_mx_bay.pkl"
else:
    raise ValueError(f"不支持的数据集: {DATASET}，请选择 METR-LA 或 PEMS-BAY")

# ======================== 路径 =========================
DATA_ROOT = f"../DATA/JC/{DATASET}"

# ======================== 日志与保存 =========================
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"./results/DCRNN_{DATASET}_{run_time}"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "log.txt")
EXCEL_FILE = os.path.join(SAVE_DIR, "metrics.xlsx")
PLOT_FILE = os.path.join(SAVE_DIR, "training_curves.png")

# 记录训练曲线
history = {
    "epoch": [],
    "train_loss": [],
    "val_mae": [],
    "val_rmse": [],
    "val_mape": []
}


def log_print(msg):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ===========================================================================
# ======================== 工具函数 ========================
# ===========================================================================
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ======================== 自动兼容二进制pickle + 文本格式 ========================
def load_graph_data(pkl_filename):
    try:
        data = np.load(pkl_filename, allow_pickle=True, encoding='bytes')
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if isinstance(data, (list, tuple)) and len(data) == 3:
            adj_mx = data[2]
        else:
            adj_mx = data
    except:
        try:
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            if isinstance(data, (list, tuple)) and len(data) == 3:
                adj_mx = data[2]
            else:
                adj_mx = data
        except:
            adj_mx = np.eye(NUM_NODES, dtype=np.float32)

    return adj_mx.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
    # 安全修复：自动变成正确的二维方阵
    if adj_mx.ndim == 1:
        adj_mx = np.eye(NUM_NODES, dtype=np.float32)
    adj_mx = adj_mx[:NUM_NODES, :NUM_NODES]
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    rw = d_mat_inv.dot(adj).tocoo()
    return torch.FloatTensor(rw.toarray().astype(np.float32)).to(DEVICE)


def load_dataset(root):
    data = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(root, f'{split}.npz')
        npz = np.load(path)
        data[f'x_{split}'] = npz['x'][..., [0]]
        data[f'y_{split}'] = npz['y'][..., [0]]
    scaler = StandardScaler(data['x_train'][..., 0].mean(), data['x_train'][..., 0].std())
    for split in ['train', 'val', 'test']:
        data[f'x_{split}'] = scaler.transform(data[f'x_{split}'])
        data[f'y_{split}'] = scaler.transform(data[f'y_{split}'])
    return data, scaler


# ===========================================================================
# ======================== DCRNN 模型 ========================
# ===========================================================================
class DiffusionConv(nn.Module):
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        self.W = nn.Linear(in_dim * (2 * K + 1), out_dim)

    def forward(self, x, supports):
        x_list = [x]
        for adj in supports:
            xp = x
            for _ in range(self.K):
                xp = torch.matmul(adj, xp)
                x_list.append(xp)
        return self.W(torch.cat(x_list, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, supports, K):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = DiffusionConv(in_dim + hidden_dim, hidden_dim * 2, K)
        self.cand_conv = DiffusionConv(in_dim + hidden_dim, hidden_dim, K)
        self.supports = supports

    def forward(self, x, h):
        comb = torch.cat([x, h], -1)
        r, u = torch.chunk(torch.sigmoid(self.gate_conv(comb, self.supports)), 2, -1)
        c = torch.tanh(self.cand_conv(torch.cat([x, r * h], -1), self.supports))
        return u * h + (1 - u) * c


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, supports, K):
        super().__init__()
        self.layers = nn.ModuleList([DCGRUCell(in_dim, hidden_dim, supports, K)])
        for _ in range(num_layers - 1):
            self.layers.append(DCGRUCell(hidden_dim, hidden_dim, supports, K))

    def forward(self, x_seq):
        B, T, N, D = x_seq.shape
        h = [torch.zeros(B, N, self.layers[0].hidden_dim).to(DEVICE) for _ in self.layers]
        for t in range(T):
            x = x_seq[:, t]
            for i, layer in enumerate(self.layers):
                h[i] = layer(x, h[i])
                x = h[i]
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, supports, K):
        super().__init__()
        self.layers = nn.ModuleList([DCGRUCell(out_dim, hidden_dim, supports, K)])
        for _ in range(num_layers - 1):
            self.layers.append(DCGRUCell(hidden_dim, hidden_dim, supports, K))
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, h_enc, pred_len):
        B, N, _ = h_enc[0].shape
        out = []
        x = torch.zeros(B, N, OUTPUT_DIM).to(DEVICE)
        h = h_enc
        for _ in range(pred_len):
            for i, layer in enumerate(self.layers):
                h[i] = layer(x, h[i])
                x = h[i]
            x = self.proj(x)
            out.append(x.unsqueeze(1))
        return torch.cat(out, dim=1)


class DCRNN(nn.Module):
    def __init__(self, supports):
        super().__init__()
        self.enc = Encoder(INPUT_DIM, RNN_UNITS, NUM_LAYERS, supports, MAX_DIFFUSION_STEP)
        self.dec = Decoder(RNN_UNITS, OUTPUT_DIM, NUM_LAYERS, supports, MAX_DIFFUSION_STEP)

    def forward(self, x):
        return self.dec(self.enc(x), PRED_LEN)


# ===========================================================================
# ======================== 指标 ========================
# ===========================================================================
def masked_mae(pred, label, scaler):
    pred = scaler.inverse_transform(pred)
    label = scaler.inverse_transform(label)
    mask = (label > 0).float()
    loss = F.l1_loss(pred, label, reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def masked_rmse(pred, label, scaler):
    pred = scaler.inverse_transform(pred)
    label = scaler.inverse_transform(label)
    mask = (label > 0).float()
    loss = (pred - label) ** 2
    return torch.sqrt(((loss * mask).sum() / (mask.sum() + 1e-8)))


def masked_mape(pred, label, scaler):
    pred = scaler.inverse_transform(pred)
    label = scaler.inverse_transform(label)
    mask = (label > 1).float()
    loss = torch.abs((pred - label) / (label + 1e-8))
    return (loss * mask).sum() / (mask.sum() + 1e-8) * 100


def evaluate_val_full(model, x, y, scaler, batch_size):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            pb = model(xb)
            total_mae += masked_mae(pb, yb, scaler).item() * len(xb)
            total_rmse += masked_rmse(pb, yb, scaler).item() * len(xb)
            total_mape += masked_mape(pb, yb, scaler).item() * len(xb)
            cnt += len(xb)
    return total_mae / cnt, total_rmse / cnt, total_mape / cnt


def evaluate_test(model, x, y, scaler, batch_size):
    model.eval()
    mae = [0., 0., 0.]
    rmse = [0., 0., 0.]
    mape = [0., 0., 0.]
    h_list = [2, 5, 11]
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            pb = model(xb)
            for j, h in enumerate(h_list):
                mae[j] += masked_mae(pb[:, h], yb[:, h], scaler).item() * len(xb)
                rmse[j] += masked_rmse(pb[:, h], yb[:, h], scaler).item() * len(xb)
                mape[j] += masked_mape(pb[:, h], yb[:, h], scaler).item() * len(xb)
            cnt += len(xb)
    return [m / cnt for m in mae], [r / cnt for r in rmse], [p / cnt for p in mape]


# ===========================================================================
# ======================== 绘图 + 保存Excel ========================
# ===========================================================================
def save_plots():
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    epochs = history['epoch']
    axs[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axs[0].set_title('Training Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, history['val_mae'], label='Val MAE', color='orange', linewidth=2)
    axs[1].set_title('Validation MAE')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(epochs, history['val_rmse'], label='Val RMSE', color='green', linewidth=2)
    axs[2].set_title('Validation RMSE')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(epochs, history['val_mape'], label='Val MAPE(%)', color='red', linewidth=2)
    axs[3].set_title('Validation MAPE')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"✅ 训练曲线已保存：{PLOT_FILE}")


def save_excel(test_mae, test_rmse, test_mape):
    # 训练日志
    df_train = pd.DataFrame(history)

    # 测试结果
    test_data = {
        '预测步长': ['15min', '30min', '60min'],
        'MAE': test_mae,
        'RMSE': test_rmse,
        'MAPE(%)': test_mape
    }
    df_test = pd.DataFrame(test_data)

    with pd.ExcelWriter(EXCEL_FILE) as writer:
        df_train.to_excel(writer, sheet_name='训练日志', index=False)
        df_test.to_excel(writer, sheet_name='测试结果', index=False)

    log_print(f"✅ 所有指标已保存到 Excel：{EXCEL_FILE}")


# ===========================================================================
# ======================== 训练 ========================
# ===========================================================================
if __name__ == "__main__":
    adj = load_graph_data(ADJ_PATH)
    supports = [calculate_random_walk_matrix(adj), calculate_random_walk_matrix(adj.T)]
    data, scaler = load_dataset(DATA_ROOT)

    x_train = torch.FloatTensor(data['x_train']).to(DEVICE)
    y_train = torch.FloatTensor(data['y_train']).to(DEVICE)
    x_val = torch.FloatTensor(data['x_val']).to(DEVICE)
    y_val = torch.FloatTensor(data['y_val']).to(DEVICE)
    x_test = torch.FloatTensor(data['x_test']).to(DEVICE)
    y_test = torch.FloatTensor(data['y_test']).to(DEVICE)

    model = DCRNN(supports).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_mae = 1e9
    wait = 0

    log_print(f"===== DCRNN 训练 - {DATASET} 数据集 =====")
    log_print(f"结果保存到: {SAVE_DIR}")
    log_print(f"节点数: {NUM_NODES}")
    log_print(f"邻接矩阵: {ADJ_PATH}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(range(0, len(x_train), BATCH_SIZE), desc=f"Epoch {epoch}/{EPOCHS}")

        for i in pbar:
            xb = x_train[i:i + BATCH_SIZE]
            yb = y_train[i:i + BATCH_SIZE]
            pb = model(xb)
            loss = masked_mae(pb, yb, scaler)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total_loss += loss.item() * len(xb)
            pbar.set_postfix({'loss': f'{loss.item():.2f}'})

        train_loss = total_loss / len(x_train)
        val_mae, val_rmse, val_mape = evaluate_val_full(model, x_val, y_val, scaler, BATCH_SIZE)

        # 记录历史
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_mape'].append(val_mape)

        log_print(
            f"[Epoch {epoch:2d}] Train Loss: {train_loss:.2f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | Val MAPE: {val_mape:.2f}%")

        if val_mae < best_mae:
            best_mae = val_mae
            wait = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
            log_print("✅ 保存最优模型")
        else:
            wait += 1
            log_print(f"Early stop counter: {wait}/{PATIENCE}")
            if wait >= PATIENCE:
                log_print("🛑 早停！")
                break

    # 绘图
    save_plots()

    # 测试
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best.pth")))
    mae, rmse, mape = evaluate_test(model, x_test, y_test, scaler, BATCH_SIZE)

    log_print("\n" + "=" * 50)
    log_print(f"          DCRNN {DATASET} 测试结果")
    log_print("=" * 50)
    log_print(f"15min  MAE: {mae[0]:.2f} | RMSE: {rmse[0]:.2f} | MAPE: {mape[0]:.2f}%")
    log_print(f"30min  MAE: {mae[1]:.2f} | RMSE: {rmse[1]:.2f} | MAPE: {mape[1]:.2f}%")
    log_print(f"60min  MAE: {mae[2]:.2f} | RMSE: {rmse[2]:.2f} | MAPE: {mape[2]:.2f}%")
    log_print("=" * 50)

    # 保存Excel
    save_excel(mae, rmse, mape)