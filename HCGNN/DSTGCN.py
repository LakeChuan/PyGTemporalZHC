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
        self.log.flush()  # 实时写入

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

# ====================== 图加载（你原来的路径！） ======================
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

class FloodDataset(Dataset):
    def __init__(self, split="train", args=None):
        self.split = split
        self.args = args
        self.samples = []

        # 逐通道归一化参数
        self.data_min = None
        self.data_max = None
        self.label_min = None
        self.label_max = None

    # 先设置全局统计量，再加载数据
    def set_global_stats(self, data_min, data_max, label_min, label_max):
        self.data_min = data_min
        self.data_max = data_max
        self.label_min = label_min
        self.label_max = label_max

    # 单独调用加载数据
    def load_data(self):
        args = self.args
        all_data = []
        all_labels = []

        for loc in args.locations:
            base = os.path.join(args.data_root, self.split, loc)
            static = np.load(os.path.join(base, "static", "static_features_80x80x3.npy"))  # [H, W, 3]
            rain_root = os.path.join(base, "dynamic", "rain")

            # 只遍历文件夹
            for name in tqdm([f for f in os.listdir(rain_root) if os.path.isdir(os.path.join(rain_root, f))],
                             desc=f"读取 {self.split}"):
                rp = os.path.join(rain_root, name)
                if not os.path.isdir(rp):
                    continue

                # 列出目录下所有文件
                files = os.listdir(rp)

                # 没有任何文件 → 直接跳过
                if not files:
                    continue
                # 找是否存在 _feature.npy 文件
                npys = [f for f in os.listdir(rp) if f.endswith("_feature.npy")]
                # 没有 npy 文件 → 跳过
                if not npys:
                    continue

                # ✅ 我的数据：dyn = [T, H, W, C]
                dyn = np.load(os.path.join(rp, npys[0]))
                T, H, W, C = dyn.shape

                # ✅ 静态特征扩展
                sta = np.tile(static[None, ...], (T, 1, 1, 1))  # [T, H, W, 3]

                # ✅ 拼接
                fus = np.concatenate([dyn, sta], axis=-1)
                fus = fus.reshape(T, H * W, 5)

                lab = dyn[..., 1:2].reshape(T, H * W, 1)

                all_data.append(fus)
                all_labels.append(lab)

        # ===================== 训练集自己计算统计量 =====================
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

        # ===================== 滑窗 + 归一化 =====================
        iw, ow = args.time_steps, args.output_steps
        eps = 1e-8

        for fus, lab in zip(all_data, all_labels):
            T = fus.shape[0]
            if T < iw + ow:
                continue

            # 逐通道归一化
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


# ====================== DSTGCN 网络（你原来的） ======================
class TemporalConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=(3,1), padding=(1,0))
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(cout)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        x = self.relu(x)
        # x = self.relu(self.conv(x))
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
    # 接收 args 做参数化，更规范更灵活
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.st1 = STBlock(5, 64)
        self.st2 = STBlock(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.st1(x, edge_index, edge_weight)
        x = self.st2(x, edge_index, edge_weight)
        x = self.out(x)

        # ✅ 从 args 取预测步长
        return x[:, :self.args.output_steps]

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

    # 1. Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'b-o', label='训练 Loss', linewidth=2, markersize=4)
    plt.title('Loss 变化曲线', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. MAE
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_mae, 'r-o', label='训练 MAE', linewidth=2, markersize=4)
    plt.plot(epochs, val_mae, 'g-o', label='验证 MAE', linewidth=2, markersize=4)
    plt.title('MAE 变化曲线', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. RMSE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_rmse, 'orange', marker='o', label='训练 RMSE', linewidth=2, markersize=4)
    plt.plot(epochs, val_rmse, 'purple', marker='o', label='验证 RMSE', linewidth=2, markersize=4)
    plt.title('RMSE 变化曲线', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. NSE
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_nse, 'blue', marker='o', label='训练 NSE', linewidth=2, markersize=4)
    plt.plot(epochs, val_nse, 'red', marker='o', label='验证 NSE', linewidth=2, markersize=4)
    plt.axhline(0, c='gray', ls='--', alpha=0.6)
    plt.title('NSE 变化曲线', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('NSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, "训练指标曲线.png")  # 这里改了
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_result_with_diff(y, pred, args):  # 加入 args
    y = y[0,0].reshape(80,80).cpu().numpy()
    p = pred[0,0].reshape(80,80).cpu().numpy()
    diff = y - p

    vmin, vmax = np.min([y, p]), np.max([y, p])
    diff_norm = TwoSlopeNorm(vcenter=0)

    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131); im1=ax1.imshow(y, cmap='Blues', vmin=vmin, vmax=vmax); ax1.set_title("真实水深"); ax1.axis("off"); plt.colorbar(im1, ax=ax1, shrink=0.75)
    ax2 = plt.subplot(132); im2=ax2.imshow(p, cmap='Blues', vmin=vmin, vmax=vmax); ax2.set_title("预测水深"); ax2.axis("off"); plt.colorbar(im2, ax=ax2, shrink=0.75)
    ax3 = plt.subplot(133); im3=ax3.imshow(diff, cmap='RdBu_r', norm=diff_norm); ax3.set_title("预测误差"); ax3.axis("off"); plt.colorbar(im3, ax=ax3, shrink=0.75)

    plt.tight_layout()
    # 这里替换成 args.save_dir
    save_path = os.path.join(args.save_dir, "水深预测+误差图.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run(args):

    # ===================== 早停初始化 =====================
    early_stopping = EarlyStopping(patience=5, verbose=True)  # 连续5轮不提升就停

    # ========== 自动创建时间戳文件夹 ==========
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"run_result_{args.modelname}/{timestamp}_DSTGCN_Epoch{args.epochs}"
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir  # 把保存路径也塞进 args
    print(f"\n📂 本次运行结果保存到: {save_dir}\n")

    # ========== 在这里绑定日志到 save_dir ==========
    log_path = os.path.join(save_dir, "training_log.txt")
    sys.stdout = Logger(log_path)
    # ================================================

    print(f"\n🚀 启动 {args.modelname}训练\n")

    # 1. 创建数据集
    train_dataset = FloodDataset("train", args)
    val_dataset = FloodDataset("val", args)
    test_dataset = FloodDataset("test", args)

    # 2. 训练集先加载 → 计算出统计量
    train_dataset.load_data()

    # 3. 把统计量设置给 val 和 test
    val_dataset.set_global_stats(
        train_dataset.data_min, train_dataset.data_max,
        train_dataset.label_min, train_dataset.label_max
    )
    test_dataset.set_global_stats(
        train_dataset.data_min, train_dataset.data_max,
        train_dataset.label_min, train_dataset.label_max
    )

    # 4. 再加载 val 和 test 的数据
    val_dataset.load_data()
    test_dataset.load_data()

    # 5. 构建 DataLoader
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)


    # 加载图结构
    ei, ew = load_graph(os.path.join(args.data_root, "train", args.locations[0], "graph"))
    ei, ew = ei.to(args.device), ew.to(args.device)

    # 模型
    model = DSTGCN(args).to(args.device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_rmse = 1e9

    history = {
        "train_loss": [],
        "val_loss": [],


        "train_mae": [],
        "train_rmse": [],
        "train_nse": [],

        "val_mae": [],
        "val_rmse": [],
        "val_nse": [],
    }

    for epoch in range(args.epochs):
        print(f"\n==================================================")
        print(f"📌 Epoch {epoch+1}/{args.epochs}")
        print(f"==================================================")

        # 训练
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"训练中", ncols=90)
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, ei, ew)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = total_loss / len(train_loader.dataset)

        # print(f"\n训练集 | Loss {train_loss:.4f} ")

        # 训练集评估
        model.eval()
        tr_mae, tr_rmse, tr_nse = 0, 0, 0
        pbar_tr = tqdm(train_loader, desc=f"训练集评估", ncols=90)
        with torch.no_grad():
            for x,y in pbar_tr:
                x,y = x.to(args.device), y.to(args.device)
                pred = model(x,ei,ew)
                mae, rmse, nse, r2 = calc_metrics(y,pred)
                tr_mae += mae*x.size(0)
                tr_rmse += rmse*x.size(0)
                tr_nse += nse*x.size(0)
        tr_mae /= len(train_loader.dataset)
        tr_rmse /= len(train_loader.dataset)
        tr_nse /= len(train_loader.dataset)

        print(f"\n训练集 | Loss {train_loss:.4f} | MAE {tr_mae:.4f} | RMSE {tr_rmse:.4f} | NSE {tr_nse:.4f}")

        # 验证集评估
        val_mae, val_rmse, val_nse = 0, 0, 0
        pbar_val = tqdm(val_loader, desc=f"验证集评估", ncols=90)
        with torch.no_grad():
            for x,y in pbar_val:
                x,y = x.to(args.device), y.to(args.device)
                pred = model(x,ei,ew)
                mae, rmse, nse, r2 = calc_metrics(y,pred)
                val_mae += mae*x.size(0)
                val_rmse += rmse*x.size(0)
                val_nse += nse*x.size(0)
        val_mae /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        val_nse /= len(val_loader.dataset)

        print(f"验证集 | MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | NSE {val_nse:.4f}")

        # 记录
        history["train_loss"].append(train_loss)
        history["train_mae"].append(tr_mae)
        history["train_rmse"].append(tr_rmse)
        history["train_nse"].append(tr_nse)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_nse"].append(val_nse)

        # 保存最优模型
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_dstgcn.pth"))
            print(f"✅ 第{epoch}轮验证集最优 → 模型已保存")

        scheduler.step(val_rmse)

        # ============== 早停检查 ==============
        early_stopping(val_rmse, model, args)  # 用 RMSE 监控
        if early_stopping.early_stop:
            print("🚨 连续 5 轮无提升，训练提前停止！")
            break

    # 测试
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_dstgcn.pth")))
    model.eval()
    print("\n🎉 测试集结果：")

    with torch.no_grad():
        total_mae = 0
        total_rmse = 0
        total_nse = 0
        total_r2 = 0
        count = 0

        # 正常跑完全部测试集，进度条会正常动
        pbar = tqdm(test_loader, desc="测试集进度", ncols=90)
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, ei, ew)
            mae, rmse, nse, r2 = calc_metrics(y, pred)

            total_mae += mae * x.size(0)
            total_rmse += rmse * x.size(0)
            total_nse += nse * x.size(0)
            total_r2 += r2 * x.size(0)
            count += x.size(0)

        # 最后统一打印平均指标
        avg_mae = total_mae / count
        avg_rmse = total_rmse / count
        avg_nse = total_nse / count
        avg_r2 = total_r2 / count

        print(f"\n测试集最终指标：")
        print(f"MAE  {avg_mae:.4f}")
        print(f"RMSE {avg_rmse:.4f}")
        print(f"NSE  {avg_nse:.4f}")
        print(f"R²   {avg_r2:.4f}")

        # 只画第一张图（不影响进度条）
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, ei, ew)
            plot_result_with_diff(y, pred, args)
            break

    # 保存参数
    with open(os.path.join(save_dir, "运行参数.txt"), "w", encoding="utf-8") as f:
        f.write(str(vars(args)))  # 把 args 转成字典保存

    # 绘图
    print("\n📈 绘制指标曲线...")
    plot_and_save_metrics(
        history["train_loss"],
        history["train_mae"],
        history["val_mae"],
        history["train_rmse"],
        history["val_rmse"],
        history["train_nse"],
        history["val_nse"],
        args
    )

    # ===================== 保存历史到 Excel =====================
    print("\n📊 保存训练历史到 Excel...")
    history_df = pd.DataFrame({
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "train_mae": history["train_mae"],
        "val_mae": history["val_mae"],
        "train_rmse": history["train_rmse"],
        "val_rmse": history["val_rmse"],
        "train_nse": history["train_nse"],
        "val_nse": history["val_nse"],
    })
    excel_path = os.path.join(args.save_dir, "训练历史.xlsx")
    history_df.to_excel(excel_path, index=False)
    print(f"✅ 训练历史已保存到：\n{excel_path}")

    print(f"\n✅ 全部完成！所有结果保存在：\n{save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="水文时空预测模型")

    # 模型参数
    parser.add_argument('--modelname', type=str, default="DSTGCN", help='模型名字DSTGCN')
    parser.add_argument('--time_steps', type=int, default=24, help='历史输入时间步')
    parser.add_argument('--output_steps', type=int, default=12, help='预测时间步')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')

    # 固定配置（也放进 args，统一管理）
    parser.add_argument('--grid_size', type=int, default=80)
    parser.add_argument('--node_num', type=int, default=80*80)
    parser.add_argument('--data_root', type=str, default="./data")
    parser.add_argument('--locations', nargs='+', default=["loaction1"])  # 列表参数

    # 设备自动判断
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 标准化开关
    parser.add_argument('--normalize', type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 打印所有参数
    print("\n✅ 加载参数: ")
    for k, v in vars(args).items():
        print(f"   {k:<18} = {v}")

    # 显式传参！最规范
    run(args)