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

# ======================== 超参数（稳定版）=========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 12
PRED_LEN = 12
NUM_NODES = 207
INPUT_DIM = 1
OUTPUT_DIM = 1
RNN_UNITS = 64        # 恢复
NUM_LAYERS = 2        # 恢复
MAX_DIFFUSION_STEP = 2
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001             # 变小，训练更稳定
USE_CURRICULUM = False # 关闭课程学习，防止震荡
CL_DECAY_STEPS = 1000

# ======================== 路径 =========================
DATA_ROOT = "./METR-LA"
ADJ_PATH = "./sensor_graph/adj_mx.pkl"

# ======================== 时间戳目录 =========================
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"./results/DCRNN_METR-LA_{run_time}"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "log.txt")

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

def load_graph_data(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    return adj_mx.astype(np.float32)

def calculate_random_walk_matrix(adj_mx):
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    rw = d_mat_inv.dot(adj).tocoo()
    return torch.FloatTensor(rw.toarray()).to(DEVICE)

def load_dataset(root):
    data = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(root, f'{split}.npz')
        npz = np.load(path)
        data[f'x_{split}'] = npz['x'][..., [0]]
        data[f'y_{split}'] = npz['y'][..., [0]]
    scaler = StandardScaler(data['x_train'][...,0].mean(), data['x_train'][...,0].std())
    for split in ['train', 'val', 'test']:
        data[f'x_{split}'] = scaler.transform(data[f'x_{split}'])
        data[f'y_{split}'] = scaler.transform(data[f'y_{split}'])
    return data, scaler

# ===========================================================================
# ======================== 扩散卷积 ========================
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

# ===========================================================================
# ======================== DCGRU Cell ========================
# ===========================================================================
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

# ===========================================================================
# ======================== Encoder / Decoder ========================
# ===========================================================================
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, supports, K):
        super().__init__()
        self.layers = nn.ModuleList([DCGRUCell(in_dim, hidden_dim, supports, K)])
        for _ in range(num_layers-1):
            self.layers.append(DCGRUCell(hidden_dim, hidden_dim, supports, K))
    def forward(self, x_seq):
        B, T, N, D = x_seq.shape
        h = [torch.zeros(B, N, self.layers[0].hidden_dim).to(DEVICE) for _ in self.layers]
        for t in range(T):
            x = x_seq[:,t]
            for i, layer in enumerate(self.layers):
                h[i] = layer(x, h[i])
                x = h[i]
        return h

class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, supports, K):
        super().__init__()
        self.layers = nn.ModuleList([DCGRUCell(out_dim, hidden_dim, supports, K)])
        for _ in range(num_layers-1):
            self.layers.append(DCGRUCell(hidden_dim, hidden_dim, supports, K))
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.supports = supports
    def forward(self, h_enc, y_gt, pred_len, use_cur, step):
        B, N, _ = h_enc[0].shape
        out, x = [], torch.zeros(B, N, OUTPUT_DIM).to(DEVICE)
        h = h_enc
        for t in range(pred_len):
            for i, layer in enumerate(self.layers):
                h[i] = layer(x, h[i])
                x = h[i]
            x = self.proj(x)
            out.append(x.unsqueeze(1))
            if use_cur and self.training and np.random.rand() < CL_DECAY_STEPS/(CL_DECAY_STEPS+np.exp(step/CL_DECAY_STEPS)):
                x = y_gt[:,t]
        return torch.cat(out, 1)

class DCRNN(nn.Module):
    def __init__(self, supports):
        super().__init__()
        self.enc = Encoder(INPUT_DIM, RNN_UNITS, NUM_LAYERS, supports, MAX_DIFFUSION_STEP)
        self.dec = Decoder(RNN_UNITS, OUTPUT_DIM, NUM_LAYERS, supports, MAX_DIFFUSION_STEP)
    def forward(self, x, y, step=0):
        return self.dec(self.enc(x), y, PRED_LEN, USE_CURRICULUM, step)

# ===========================================================================
# ======================== 指标 ========================
# ===========================================================================
def masked_mae(pred, label, null=0):
    m = (label != null).float()
    m /= m.mean()
    return (F.l1_loss(pred, label, reduction='none') * m).mean()

def masked_rmse(pred, label, null=0):
    m = (label != null).float()
    m /= m.mean()
    return torch.sqrt((F.mse_loss(pred, label, reduction='none') * m).mean())

def masked_mape(pred, label, null=0):
    m = (label != null).float()
    m /= m.mean()
    return (torch.abs((pred-label)/label) * m).mean()

# ===========================================================================
# ======================== 流畅训练核心 ========================
# ===========================================================================
def evaluate(model, x, y, batch_size):
    model.eval()
    total_mae = 0
    total_rmse = 0
    total_mape = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            x_b = x[i:i+batch_size]
            y_b = y[i:i+batch_size]
            pred = model(x_b, y_b)
            total_mae += masked_mae(pred, y_b).item() * len(x_b)
            total_rmse += masked_rmse(pred, y_b).item() * len(x_b)
            total_mape += masked_mape(pred, y_b).item() * len(x_b)
            count += len(x_b)
    return total_mae/count, total_rmse/count, total_mape/count

# ===========================================================================
# ======================== 主程序 ========================
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

    train_losses, val_maes = [], []
    best_mae = 1e9

    log_print(f"===== DCRNN Training Started at {run_time} =====")
    log_print(f"Save directory: {SAVE_DIR}")

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        pbar = tqdm(range(0, len(x_train), BATCH_SIZE), desc=f"Epoch {epoch}/{EPOCHS}")

        for i in pbar:
            x = x_train[i:i+BATCH_SIZE]
            y = y_train[i:i+BATCH_SIZE]
            pred = model(x, y, epoch)
            loss = masked_mae(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(x_train)
        train_losses.append(avg_loss)

        # ========== 【流畅验证】关键优化 ==========
        mae_v, rmse_v, mape_v = evaluate(model, x_val, y_val, BATCH_SIZE)
        val_maes.append(mae_v)

        log_print(f"[Epoch {epoch:2d}] TrainLoss: {avg_loss:.4f} | Val MAE: {mae_v:.4f} RMSE: {rmse_v:.4f} MAPE: {mape_v:.4f}")

        if mae_v < best_mae:
            best_mae = mae_v
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

    # 测试
    mae_t, rmse_t, mape_t = evaluate(model, x_test, y_test, BATCH_SIZE)
    log_print("\n===== FINAL TEST RESULT =====")
    log_print(f"MAE:  {mae_t:.4f}")
    log_print(f"RMSE: {rmse_t:.4f}")
    log_print(f"MAPE: {mape_t:.4f}")

    # 绘图
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss')
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(val_maes, label='Val MAE', color='orange')
    plt.title('Val MAE')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "curves.png"), dpi=150)
    plt.show()

    log_print(f"\nAll results saved to: {SAVE_DIR}")