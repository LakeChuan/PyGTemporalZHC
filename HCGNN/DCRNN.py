import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.nn import ChebConv
import matplotlib.pyplot as plt
import time
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================== 日志保存到 save_dir ========================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, args):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⏸ 早停计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, args):
        if self.verbose:
            print(f'✅ 验证集性能提升，保存最优模型 (验证损失: {val_loss:.6f})')
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))


# ====================== 图加载 ======================
def load_graph(graph_dir):
    edge_index, edge_weight = [], []
    txt_path = os.path.join(graph_dir, "directed_improved_d8_graph.txt")
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or "from_node" in line:
                continue
            u, v, w = line.split()
            edge_index.append([int(u), int(v)])
            edge_weight.append(float(w))
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    ew = torch.tensor(edge_weight, dtype=torch.float)
    return ei, ew


class FloodDataset(Dataset):
    def __init__(self, split="train", args=None):
        self.split = split
        self.args = args
        self.samples = []
        self.data_min = None
        self.data_max = None
        self.label_min = None
        self.label_max = None

    def set_global_stats(self, data_min, data_max, label_min, label_max):
        self.data_min = data_min
        self.data_max = data_max
        self.label_min = label_min
        self.label_max = label_max

    def load_data(self):
        args = self.args
        all_data = []
        all_labels = []

        for loc in args.locations:
            base = os.path.join(args.data_root, self.split, loc)
            static = np.load(os.path.join(base, "static", "static_features_80x80x3.npy"))
            rain_root = os.path.join(base, "dynamic", "rain")

            for name in tqdm([f for f in os.listdir(rain_root) if os.path.isdir(os.path.join(rain_root, f))],
                             desc=f"读取 {self.split}"):
                rp = os.path.join(rain_root, name)
                npys = [f for f in os.listdir(rp) if f.endswith("_feature.npy")]
                if not npys:
                    continue

                dyn = np.load(os.path.join(rp, npys[0]))
                T, H, W, C = dyn.shape
                sta = np.tile(static[None, ...], (T, 1, 1, 1))
                fus = np.concatenate([dyn, sta], axis=-1)
                fus = fus.reshape(T, H * W, 5)
                lab = dyn[..., 1:2].reshape(T, H * W, 1)
                all_data.append(fus)
                all_labels.append(lab)

        if self.split == "train" and len(all_data) > 0:
            data_stack = np.concatenate(all_data, axis=0)
            self.data_min = data_stack.min(axis=(0, 1))
            self.data_max = data_stack.max(axis=(0, 1))
            label_stack = np.concatenate(all_labels, axis=0)
            self.label_min = label_stack.min()
            self.label_max = label_stack.max()

            print("\n📊 全局逐通道归一化参数：")
            for i in range(len(self.data_min)):
                print(f"  通道 {i} | min={self.data_min[i]:>6.3f} | max={self.data_max[i]:>6.3f}")
            print(f"  标签    | min={self.label_min:>6.3f} | max={self.label_max:>6.3f}\n")

        iw, ow = args.time_steps, args.output_steps
        eps = 1e-8
        for fus, lab in zip(all_data, all_labels):
            T = fus.shape[0]
            if T < iw + ow:
                continue
            fus = (fus - self.data_min) / (self.data_max - self.data_min + eps)
            lab = (lab - self.label_min) / (self.label_max - self.label_min + eps)
            for i in range(T - iw - ow + 1):
                self.samples.append((fus[i:i+iw], lab[i+iw:i+iw+ow]))

        print(f"✅ {self.split:5} 样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ====================== 【终极修复版】DCRNN 模型 ======================
from torch_geometric.utils import add_self_loops, degree

class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=1):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        B, N, C = x.shape
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        row, col = edge_index
        deg = degree(col, N, dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[col]

        x_t = x.transpose(0, 1)
        x_t = torch.sparse.mm(torch.sparse_coo_tensor(edge_index, norm, (N, N)), x_t)
        out = x_t.transpose(0, 1)
        return self.lin(out)


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_reset = DiffusionConv(input_dim + hidden_dim, hidden_dim)
        self.conv_update = DiffusionConv(input_dim + hidden_dim, hidden_dim)
        self.conv_candidate = DiffusionConv(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, hidden, edge_index, edge_weight):
        combined = torch.cat([x, hidden], dim=-1)
        reset = torch.sigmoid(self.conv_reset(combined, edge_index, edge_weight))
        update = torch.sigmoid(self.conv_update(combined, edge_index, edge_weight))
        candidate = torch.tanh(self.conv_candidate(torch.cat([x, reset * hidden], dim=-1), edge_index, edge_weight))
        return update * hidden + (1 - update) * candidate


class DCRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_dim = 5
        self.hidden_dim = 32

        self.encoder = DCGRUCell(self.input_dim, self.hidden_dim)
        self.decoder = DCGRUCell(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        B, T_in, N, C = x.shape
        T_out = self.args.output_steps

        hidden = torch.zeros(B, N, self.hidden_dim, device=x.device)
        for t in range(T_in):
            hidden = self.encoder(x[:, t], hidden, edge_index, edge_weight)

        outputs = []
        decoder_input = torch.zeros(B, N, self.hidden_dim, device=x.device)
        for _ in range(T_out):
            hidden = self.decoder(decoder_input, hidden, edge_index, edge_weight)
            outputs.append(hidden.unsqueeze(1))

        return self.out(torch.cat(outputs, dim=1))


# ====================== 指标 ======================
def calc_metrics(y_true, y_pred):
    yt = y_true.flatten().cpu().numpy()
    yp = y_pred.flatten().cpu().numpy()
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mean_yt = np.mean(yt)
    nse = 1 - (np.sum((yt - yp) ** 2) / (np.sum((yt - mean_yt) ** 2) + 1e-8))
    r2 = nse
    return mae, rmse, nse, r2


def plot_and_save_metrics(train_loss, train_mae, val_mae, train_rmse, val_rmse, train_nse, val_nse, args):
    epochs = list(range(1, len(train_loss)+1))
    plt.figure(figsize=(20, 14))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'b-o')
    plt.title('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_mae, 'r-o')
    plt.plot(epochs, val_mae, 'g-o')
    plt.title('MAE')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_rmse, 'orange-o')
    plt.plot(epochs, val_rmse, 'purple-o')
    plt.title('RMSE')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_nse, 'b-o')
    plt.plot(epochs, val_nse, 'r-o')
    plt.title('NSE')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "metrics.png"), dpi=300)
    plt.close()


def plot_result_with_diff(y, pred, args):
    y = y[0,0].reshape(80,80).cpu().numpy()
    p = pred[0,0].reshape(80,80).cpu().numpy()
    plt.figure(figsize=(12,4))
    plt.subplot(131); plt.imshow(y); plt.title('True')
    plt.subplot(132); plt.imshow(p); plt.title('Pred')
    plt.subplot(133); plt.imshow(y-p); plt.title('Diff')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "result.png"), dpi=200)
    plt.close()


# ====================== 训练 ======================
def run(args):
    early_stopping = EarlyStopping(patience=5, verbose=True)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"run_result_{args.modelname}/{timestamp}_DCRNN"
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    log_path = os.path.join(save_dir, "training_log.txt")
    sys.stdout = Logger(log_path)

    train_dataset = FloodDataset("train", args)
    val_dataset = FloodDataset("val", args)
    test_dataset = FloodDataset("test", args)
    train_dataset.load_data()
    val_dataset.set_global_stats(train_dataset.data_min, train_dataset.data_max, train_dataset.label_min, train_dataset.label_max)
    test_dataset.set_global_stats(train_dataset.data_min, train_dataset.data_max, train_dataset.label_min, train_dataset.label_max)
    val_dataset.load_data()
    test_dataset.load_data()

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    ei, ew = load_graph(os.path.join(args.data_root, "train", args.locations[0], "graph"))
    ei, ew = ei.to(args.device), ew.to(args.device)

    model = DCRNN(args).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "train_mae": [], "train_rmse": [], "train_nse": [],
               "val_mae": [], "val_rmse": [], "val_nse": []}

    for epoch in range(args.epochs):
        print(f"\n==================================================")
        print(f"📌 Epoch {epoch+1}/{args.epochs}")
        print(f"==================================================")

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc="训练中", ncols=90)
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, ei, ew)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = total_loss / len(train_dataset)

        model.eval()
        tr_mae = tr_rmse = tr_nse = 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                mae, rmse, nse, _ = calc_metrics(y, model(x, ei, ew))
                tr_mae += mae * x.size(0)
                tr_rmse += rmse * x.size(0)
                tr_nse += nse * x.size(0)
        tr_mae /= len(train_dataset)
        tr_rmse /= len(train_dataset)
        tr_nse /= len(train_dataset)

        val_mae = val_rmse = val_nse = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                mae, rmse, nse, _ = calc_metrics(y, model(x, ei, ew))
                val_mae += mae * x.size(0)
                val_rmse += rmse * x.size(0)
                val_nse += nse * x.size(0)
        val_mae /= len(val_dataset)
        val_rmse /= len(val_dataset)
        val_nse /= len(val_dataset)

        print(f"训练 | Loss={train_loss:.4f} MAE={tr_mae:.4f} RMSE={tr_rmse:.4f} NSE={tr_nse:.4f}")
        print(f"验证 | MAE={val_mae:.4f} RMSE={val_rmse:.4f} NSE={val_nse:.4f}")

        history["train_loss"].append(train_loss)
        history["train_mae"].append(tr_mae)
        history["train_rmse"].append(tr_rmse)
        history["train_nse"].append(tr_nse)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_nse"].append(val_nse)

        torch.save(model.state_dict(), os.path.join(save_dir, "best_dcrnn.pth"))
        scheduler.step(val_rmse)
        early_stopping(val_rmse, model, args)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_dcrnn.pth")))
    model.eval()
    test_mae = test_rmse = test_nse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            mae, rmse, nse, _ = calc_metrics(y, model(x, ei, ew))
            test_mae += mae * x.size(0)
            test_rmse += rmse * x.size(0)
            test_nse += nse * x.size(0)
    test_mae /= len(test_dataset)
    test_rmse /= len(test_dataset)
    test_nse /= len(test_dataset)
    print("\n测试集结果：")
    print(f"MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | NSE: {test_nse:.4f}")

    plot_and_save_metrics(**history, args=args)
    pd.DataFrame(history).to_excel(os.path.join(save_dir, "history.xlsx"), index=False)
    print("✅ 全部完成！")


def parse_args():
    parser = argparse.ArgumentParser(description="DCRNN")
    parser.add_argument('--modelname', default="DCRNN")
    parser.add_argument('--time_steps', type=int, default=24)
    parser.add_argument('--output_steps', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--data_root', default="./data")
    parser.add_argument('--locations', nargs='+', default=["loaction1"])
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("\n✅ 加载参数: ")
    for k, v in vars(args).items():
        print(f"   {k:<18} = {v}")
    run(args)