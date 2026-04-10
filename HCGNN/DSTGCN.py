import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.nn import ChebConv
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 图加载 ======================
def load_graph(graph_dir):
    edge_index, edge_weight = [], []
    txt_path = os.path.join(graph_dir, "directed_improved_d8_graph.txt")
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or any(c in line for c in ["from_node", "to_node", "weight"]):
                continue
            u, v, w = line.split()
            edge_index.append([int(u), int(v)])
            edge_weight.append(float(w))
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    ew = torch.tensor(edge_weight, dtype=torch.float)
    return ei, ew

# ====================== 数据集 ======================
class FloodDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.samples = []
        self.load_all_data()

    def load_all_data(self):
        for loc in CONFIG["locations"]:
            base = os.path.join(CONFIG["data_root"], self.split, loc)
            static = np.load(os.path.join(base, "static", "static_features_80x80x3.npy"))
            rain_root = os.path.join(base, "dynamic", "rain")
            for name in tqdm(os.listdir(rain_root), desc=f"加载 {self.split}"):
                rp = os.path.join(rain_root, name)
                if not os.path.isdir(rp): continue
                npys = [f for f in os.listdir(rp) if f.endswith("_feature.npy")]
                if not npys: continue
                d = np.load(os.path.join(rp, npys[0]))
                H, W, C, T = d.shape
                dyn = d.transpose(3, 0, 1, 2)
                sta = np.tile(static, (T, 1, 1, 1))
                fus = np.concatenate([dyn, sta], -1).reshape(T, 6400, 5)
                lab = dyn[..., 1:2].reshape(T, 6400, 1)

                fus = (fus - fus.min()) / (fus.max() - fus.min() + 1e-8)
                lab = (lab - lab.min()) / (lab.max() - lab.min() + 1e-8)

                iw, ow = CONFIG["input_window"], CONFIG["output_window"]
                if T < iw + ow: continue
                for i in range(T - iw - ow + 1):
                    self.samples.append((fus[i:i+iw], lab[i+iw:i+iw+ow]))
        print(f"✅ {self.split:5} 样本数: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ====================== 【升级】DSTGCN 稳定网络 ======================
class TemporalConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=(3,1), padding=(1,0))
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(cout)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.relu(self.conv(x))
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        return self.drop(x)

class SpatialConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ChebConv(cin, cout, K=3)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(cout)
        self.drop = nn.Dropout(0.15)

    def forward(self, x, edge_index, edge_weight):
        x = self.relu(self.conv(x, edge_index, edge_weight))
        x = self.norm(x)
        return self.drop(x)

class STBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.tconv = TemporalConv(cin, cout)
        self.sconv = SpatialConv(cout, cout)

    def forward(self, x, edge_index, edge_weight):
        x = self.tconv(x)
        B, T, N, C = x.shape
        x = x.reshape(B*T, N, C)
        x = self.sconv(x, edge_index, edge_weight)
        x = x.reshape(B, T, N, C)
        return x

class DSTGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.st1 = STBlock(5, 64)
        self.st2 = STBlock(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.st1(x, edge_index, edge_weight)
        x = self.st2(x, edge_index, edge_weight)
        return self.out(x)[:, :CONFIG["output_window"]]

# ====================== 指标 ======================
def calc_metrics(y_true, y_pred):
    yt = y_true.flatten().cpu().numpy()
    yp = y_pred.flatten().cpu().numpy()
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mean_yt = np.mean(yt)
    nse = 1 - (np.sum((yt - yp) ** 2) / (np.sum((yt - mean_yt) ** 2) + 1e-8))
    r2 = 1 - (np.sum((yt - yp) ** 2) / (np.sum((yt - mean_yt) ** 2) + 1e-8))
    return mae, rmse, nse, r2

# ====================== 【升级】画图 + 保存 ======================
def plot_and_save_metrics(train_loss, train_rmse, val_rmse, train_nse, val_nse):
    epochs = list(range(1, len(train_loss)+1))
    plt.figure(figsize=(16,10))

    # Loss
    plt.subplot(2,2,1)
    plt.plot(epochs, train_loss, 'b-o', label='训练Loss')
    plt.title('Loss 变化曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # RMSE
    plt.subplot(2,2,2)
    plt.plot(epochs, train_rmse, 'r-o', label='训练RMSE')
    plt.plot(epochs, val_rmse, 'g-o', label='验证RMSE')
    plt.title('RMSE 变化曲线')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # NSE
    plt.subplot(2,2,3)
    plt.plot(epochs, train_nse, 'orange', marker='o', label='训练NSE')
    plt.plot(epochs, val_nse, 'purple', marker='o', label='验证NSE')
    plt.axhline(0, c='gray', ls='--')
    plt.title('NSE 变化曲线')
    plt.xlabel('Epoch')
    plt.ylabel('NSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("训练指标曲线.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_result(y, pred):
    y = y[0,0].reshape(80,80).cpu().numpy()
    p = pred[0,0].reshape(80,80).cpu().numpy()
    plt.figure(figsize=(10,4))
    plt.subplot(121); plt.imshow(y, cmap="Blues"); plt.title("真实水深"); plt.axis("off")
    plt.subplot(122); plt.imshow(p, cmap="Blues"); plt.title("预测水深"); plt.axis("off")
    plt.tight_layout()
    plt.savefig("水深预测图.png", dpi=300, bbox_inches='tight')
    plt.show()

# ====================== 【升级】训练流程（先训练 → 后验证） ======================
def run():
    print("\n🚀 启动 DSTGCN 训练（稳定增强版）\n")
    train_loader = DataLoader(FloodDataset("train"), CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(FloodDataset("val"),   CONFIG["batch_size"])
    test_loader  = DataLoader(FloodDataset("test"),  1)

    ei, ew = load_graph(os.path.join(CONFIG["data_root"], "train", CONFIG["locations"][0], "graph"))
    ei, ew = ei.to(CONFIG["device"]), ew.to(CONFIG["device"])

    model = DSTGCN().to(CONFIG["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_rmse = 1e9

    # 记录指标
    history = {
        "train_loss": [], "train_rmse": [], "train_nse": [],
        "val_rmse": [], "val_nse": []
    }

    for epoch in range(CONFIG["epochs"]):
        # ============== 1. 训练 ==============
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} 训练中")
        for x, y in pbar:
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            pred = model(x, ei, ew)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = total_loss / len(train_loader.dataset)

        # ============== 2. 训练集评估 ==============
        model.eval()
        tr_mae=tr_rmse=tr_nse=tr_r2=0
        with torch.no_grad():
            for x,y in train_loader:
                x,y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
                pred = model(x,ei,ew)
                mae,rmse,nse,r2 = calc_metrics(y,pred)
                tr_mae += mae*x.size(0)
                tr_rmse += rmse*x.size(0)
                tr_nse += nse*x.size(0)
        tr_mae /= len(train_loader.dataset)
        tr_rmse /= len(train_loader.dataset)
        tr_nse /= len(train_loader.dataset)

        print(f"\n📊 Epoch {epoch+1} 训练集结果：")
        print(f"    Loss {train_loss:.4f} | MAE {tr_mae:.4f} | RMSE {tr_rmse:.4f} | NSE {tr_nse:.4f}")

        # ============== 3. 验证集评估（你要的顺序） ==============
        print(f"🔍 开始验证集评估...")
        val_mae=val_rmse=val_nse=val_r2=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
                pred = model(x,ei,ew)
                mae,rmse,nse,r2 = calc_metrics(y,pred)
                val_mae += mae*x.size(0)
                val_rmse += rmse*x.size(0)
                val_nse += nse*x.size(0)
        val_mae /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        val_nse /= len(val_loader.dataset)

        print(f"✅ 验证集结果：")
        print(f"    MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | NSE {val_nse:.4f}")

        # 保存历史
        history["train_loss"].append(train_loss)
        history["train_rmse"].append(tr_rmse)
        history["train_nse"].append(tr_nse)
        history["val_rmse"].append(val_rmse)
        history["val_nse"].append(val_nse)

        # 保存最优模型
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), "best_dstgcn.pth")
            print("💾 验证集最优 → 模型已保存！")

        scheduler.step(val_rmse)

    # ============== 训练结束 → 画图 ==============
    print("\n📈 绘制训练指标曲线并保存...")
    plot_and_save_metrics(
        history["train_loss"],
        history["train_rmse"],
        history["val_rmse"],
        history["train_nse"],
        history["val_nse"]
    )

    # ============== 测试 ==============
    model.load_state_dict(torch.load("best_dstgcn.pth"))
    model.eval()
    print("\n🎉 测试集结果：")
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            pred = model(x,ei,ew)
            mae,rmse,nse,r2 = calc_metrics(y,pred)
            print(f"MAE {mae:.4f} | RMSE {rmse:.4f} | NSE {nse:.4f} | R² {r2:.4f}")
            plot_result(y,pred)
            break

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="DSTGCN 水文时空预测模型")
        parser.add_argument('--time_steps', type=int, default=24, help='历史输入时间步')
        parser.add_argument('--output_steps', type=int, default=12, help='预测时间步')
        parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
        parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
        parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
        return parser.parse_args()

    args = parse_args()
    CONFIG = {
        "input_window": args.time_steps,
        "output_window": args.output_steps,
        "grid_size": 80,
        "node_num": 80 * 80,
        "data_root": "./data",
        "locations": ["loaction1"],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    print("\n✅ 加载参数: ", CONFIG)
    run()