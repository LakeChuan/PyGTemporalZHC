# ==============================================
# 1. 环境导入与GPU配置
# ==============================================
import torch
import torch_geometric_temporal as pyg_temp
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import TGCN
import warnings

warnings.filterwarnings("ignore")

# 可视化库
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import gaussian_kde
import time  # 手动统计耗时

# 中文显示配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================================
# 2. GPU自动检测与配置
# ==============================================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("⚠️ 未检测到GPU，自动切换到CPU训练")

SAVE_DIR = "./pyg_temporal_plots_gpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================
# 3. 超参数配置
# ==============================================
EPOCHS = 20
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
USE_MIXED_PRECISION = True
NORMALIZE_DATA = True

print("=" * 60)
print("环境信息验证")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyG Temporal 版本: {pyg_temp.__version__}")
print(f"训练设备: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
print(f"训练参数：epochs={EPOCHS}, hidden_dim={HIDDEN_DIM}, lr={LEARNING_RATE}")
print(f"混合精度训练: {'开启' if USE_MIXED_PRECISION else '关闭'} | 数据归一化: {'开启' if NORMALIZE_DATA else '关闭'}")
print("=" * 60)

# ==============================================
# 4. 加载并预处理数据集
# ==============================================
dataset = pyg_temp.METRLADatasetLoader().get_dataset(
    num_timesteps_in=12,
    num_timesteps_out=1
)

# 数据归一化
if NORMALIZE_DATA:
    all_x = []
    all_y = []
    for batch in dataset:
        all_x.append(batch.x.cpu().numpy())
        all_y.append(batch.y.cpu().numpy())
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    x_mean = np.mean(all_x, axis=0)
    x_std = np.std(all_x, axis=0) + 1e-8
    y_mean = np.mean(all_y, axis=0)
    y_std = np.std(all_y, axis=0) + 1e-8


    class NormalizedDataset:
        def __init__(self, dataset, x_mean, x_std, y_mean, y_std):
            self.dataset = dataset
            self.x_mean = x_mean
            self.x_std = x_std
            self.y_mean = y_mean
            self.y_std = y_std

        def __iter__(self):
            for batch in self.dataset:
                x_norm = (batch.x - torch.tensor(self.x_mean).to(batch.x.device)) / torch.tensor(self.x_std).to(
                    batch.x.device)
                y_norm = (batch.y - torch.tensor(self.y_mean).to(batch.y.device)) / torch.tensor(self.y_std).to(
                    batch.y.device)
                batch.x = x_norm
                batch.y = y_norm
                yield batch


    train_dataset_raw, test_dataset_raw = temporal_signal_split(dataset, train_ratio=0.8)
    train_dataset = NormalizedDataset(train_dataset_raw, x_mean, x_std, y_mean, y_std)
    test_dataset = NormalizedDataset(test_dataset_raw, x_mean, x_std, y_mean, y_std)
else:
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)


# 统计批次数量
def count_dataset_batches(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


train_batch_count = count_dataset_batches(train_dataset)
test_batch_count = count_dataset_batches(test_dataset)
print(f"\n数据集批次数量：训练集={train_batch_count}批，测试集={test_batch_count}批")

sample = next(iter(train_dataset))
print("\n数据集基本信息：")
print(f"节点数量（监测点）: {sample.x.shape[0]}")
print(f"每个节点特征数: {sample.x.shape[1]}")
print(f"输入时间步长: {sample.x.shape[2]}")
print(f"输入维度: {sample.x.shape}, 输出维度: {sample.y.shape}")


# ==============================================
# 5. 定义TGCN模型
# ==============================================
class TGCNModel(torch.nn.Module):
    def __init__(self, node_features, time_steps, hidden_dim, output_dim):
        super(TGCNModel, self).__init__()
        self.flatten_dim = node_features * time_steps
        self.tgcn = TGCN(in_channels=self.flatten_dim, out_channels=hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x_flat = x.reshape(x.shape[0], -1)
        h = self.tgcn(x_flat, edge_index)
        h = self.dropout(h)
        out = self.linear(h)
        return out


model = TGCNModel(
    node_features=sample.x.shape[1],
    time_steps=sample.x.shape[2],
    hidden_dim=HIDDEN_DIM,
    output_dim=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler() if (USE_MIXED_PRECISION and device.type == 'cuda') else None
loss_fn = torch.nn.MSELoss()

# ==============================================
# 6. 模型训练（单行动态刷新每轮批次进度，无任何新行）
# ==============================================
import sys
import time
import os


# 适配不同系统的光标控制
def clear_line():
    """清空当前行并将光标移到行首"""
    if os.name == 'nt':  # Windows
        sys.stdout.write('\r' + ' ' * 100 + '\r')
    else:  # Linux/Mac
        sys.stdout.write('\033[K\r')
    sys.stdout.flush()


print("\n开始GPU加速训练...")
model.train()
train_start_time = time.time()

train_loss_history = []
test_loss_history = []
epochs_range = []

# 外层：总轮数进度条（固定在最上方）
epoch_pbar = tqdm(
    total=EPOCHS,
    desc="✅ 总训练轮数",
    ncols=100,
    position=0,
    dynamic_ncols=True,
    file=sys.stdout,
    leave=True
)

for epoch in range(EPOCHS):
    total_train_loss = 0.0
    batch_count = 0
    start_time = time.time()


    # 手动控制批次进度显示（核心：单行动态刷新）
    def update_batch_progress(current, total, epoch_num):
        """手动更新批次进度，只在一行显示"""
        # 计算进度百分比和耗时
        percent = current / total * 100
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0

        # 格式化进度条
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # 构造进度字符串
        progress_str = (
            f"🔄 第{epoch_num}/{EPOCHS}轮批次: {percent:5.1f}%|{bar}| "
            f"{current}/{total} [{time.strftime('%M:%S', time.gmtime(elapsed))}<"
            f"{time.strftime('%M:%S', time.gmtime(remaining))}, {rate:6.2f}it/s]"
        )

        # 清空当前行并打印新进度（核心：不换行）
        clear_line()
        sys.stdout.write(progress_str)
        sys.stdout.flush()


    # 遍历训练批次（无tqdm，纯手动控制）
    for batch in train_dataset:
        batch = batch.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                y_pred = model(batch.x, batch.edge_index)
                loss = loss_fn(y_pred, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(batch.x, batch.edge_index)
            loss = loss_fn(y_pred, batch.y)
            loss.backward()
            optimizer.step()

        total_train_loss += loss.item()
        batch_count += 1

        # 每处理10个批次更新一次进度（减少刷新频率）
        if batch_count % 10 == 0 or batch_count == train_batch_count:
            update_batch_progress(batch_count, train_batch_count, epoch + 1)

    # 轮次结束：清空进度行并打印最终状态（换行）
    clear_line()
    elapsed = time.time() - start_time
    final_rate = batch_count / elapsed if elapsed > 0 else 0
    print(
        f"🔄 第{epoch + 1}/{EPOCHS}轮批次: 100.0%|{'█' * 40}| "
        f"{batch_count}/{batch_count} [{time.strftime('%M:%S', time.gmtime(elapsed))}<00:00, {final_rate:6.2f}it/s]"
    )

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    test_batch_count = 0
    with torch.no_grad():
        for batch in test_dataset:
            batch = batch.to(device)
            y_pred = model(batch.x, batch.edge_index)
            loss = loss_fn(y_pred, batch.y)
            total_test_loss += loss.item()
            test_batch_count += 1

    avg_train_loss = total_train_loss / batch_count
    avg_test_loss = total_test_loss / test_batch_count
    model.train()

    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)
    epochs_range.append(epoch + 1)

    # 更新外层轮次进度条
    epoch_pbar.update(1)
    epoch_pbar.set_postfix({
        "训练损失": f"{avg_train_loss:.4f}",
        "测试损失": f"{avg_test_loss:.4f}"
    })
    sys.stdout.flush()

epoch_pbar.close()
train_total_time = time.time() - train_start_time
train_time_str = time.strftime("%H:%M:%S", time.gmtime(train_total_time))

# ==============================================
# 7. 绘制损失曲线
# ==============================================
fig_live, ax_live = plt.subplots(figsize=(10, 6))
ax_live.plot(epochs_range, train_loss_history, label="训练损失", color="#1f77b4", linewidth=2)
ax_live.plot(epochs_range, test_loss_history, label="测试损失", color="#ff7f0e", linewidth=2)
min_train_idx = np.argmin(train_loss_history)
min_test_idx = np.argmin(test_loss_history)
ax_live.scatter(epochs_range[min_train_idx], train_loss_history[min_train_idx],
                color="#1f77b4", s=100, zorder=5, label=f"最小训练损失: {train_loss_history[min_train_idx]:.4f}")
ax_live.scatter(epochs_range[min_test_idx], test_loss_history[min_test_idx],
                color="#ff7f0e", s=100, zorder=5, label=f"最小测试损失: {test_loss_history[min_test_idx]:.4f}")
ax_live.text(0.05, 0.85, f"训练设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
             transform=ax_live.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax_live.set_title("GPU训练 - 损失变化曲线", fontsize=14, fontweight="bold")
ax_live.set_xlabel("训练轮数")
ax_live.set_ylabel("MSE损失（归一化后）")
ax_live.legend()
ax_live.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "GPU_训练损失曲线.png"), dpi=300, bbox_inches="tight")
plt.show()

# ==============================================
# 8. 测试集预测（含进度展示）
# ==============================================
print("\n" + "=" * 60)
print("开始GPU加速测试集预测...")
model.eval()
all_preds = []
all_truths = []
test_loss_final = 0.0

test_start_time = time.time()
# 测试集预测进度条
test_pbar = tqdm(test_dataset, desc="🔍 测试集预测进度", total=test_batch_count, ncols=100)
for batch in test_pbar:
    batch = batch.to(device)
    with torch.no_grad():
        if scaler is not None:
            with torch.cuda.amp.autocast():
                y_pred = model(batch.x, batch.edge_index)
                loss = loss_fn(y_pred, batch.y)
        else:
            y_pred = model(batch.x, batch.edge_index)
            loss = loss_fn(y_pred, batch.y)

        test_loss_final += loss.item()

        if NORMALIZE_DATA:
            pred_np = y_pred.cpu().numpy() * y_std + y_mean
            truth_np = batch.y.cpu().numpy() * y_std + y_mean
        else:
            pred_np = y_pred.cpu().numpy()
            truth_np = batch.y.cpu().numpy()

        all_preds.extend(pred_np.flatten())
        all_truths.extend(truth_np.flatten())
test_total_time = time.time() - test_start_time

all_preds = np.array(all_preds)
all_truths = np.array(all_truths)
avg_test_loss_final = test_loss_final / test_batch_count

# 计算评估指标
errors = all_preds - all_truths
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
r2 = 1 - (np.sum((all_truths - all_preds) ** 2) / np.sum((all_truths - np.mean(all_truths)) ** 2))
mape = np.mean(np.abs(errors / (all_truths + 1e-8))) * 100

# 打印详细结果
print("\n" + "=" * 60)
print("GPU训练 - 模型性能详细评估结果")
print("=" * 60)
print(f"最终测试集平均MSE损失: {avg_test_loss_final:.4f}（归一化后）")
print(f"平均绝对误差 (MAE): {mae:.4f} | 均方根误差 (RMSE): {rmse:.4f}")
print(f"平均百分比误差 (MAPE): {mape:.2f}% | 决定系数 (R²): {r2:.4f}")
print(f"预测值范围: [{all_preds.min():.2f}, {all_preds.max():.2f}]")
print(f"真实值范围: [{all_truths.min():.2f}, {all_truths.max():.2f}]")
print(f"误差范围: [{errors.min():.2f}, {errors.max():.2f}] | 误差均值: {errors.mean():.4f}")
print(f"GPU训练总耗时: {train_time_str} | 测试预测耗时: {time.strftime('%H:%M:%S', time.gmtime(test_total_time))}")
print("=" * 60)

# ==============================================
# 9. 综合可视化
# ==============================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：预测值vs真实值散点图
sample_idx = np.random.choice(len(all_preds), size=1500, replace=False)
pred_sample = all_preds[sample_idx]
truth_sample = all_truths[sample_idx]
ax1.scatter(truth_sample, pred_sample, alpha=0.5, color="#2ca02c", s=15, label="预测样本")
min_val = min(truth_sample.min(), pred_sample.min())
max_val = max(truth_sample.max(), pred_sample.max())
ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="完美预测线(y=x)")
ax1.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax1.transAxes,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=12)
ax1.set_title("GPU训练 - 预测值 vs 真实值", fontsize=14, fontweight="bold")
ax1.set_xlabel("真实值（交通流量）")
ax1.set_ylabel("预测值（交通流量）")
ax1.legend()
ax1.grid(alpha=0.3)

# 子图2：误差分布直方图
ax2.hist(errors, bins=60, color="#d62728", alpha=0.7, edgecolor="black", density=True, label="误差分布")
kernel = np.linspace(errors.min(), errors.max(), 200)
kde = gaussian_kde(errors)
ax2.plot(kernel, kde(kernel), color="black", linewidth=2, label="误差密度曲线")
ax2.axvline(x=0, color="green", linestyle="--", linewidth=2, label="无误差线")
ax2.text(0.05, 0.95, f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}", transform=ax2.transAxes,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10)
ax2.set_title("GPU训练 - 预测误差分布", fontsize=14, fontweight="bold")
ax2.set_xlabel("误差值")
ax2.set_ylabel("密度")
ax2.legend()
ax2.grid(alpha=0.3)

# 子图3：前100个节点对比
ax3.plot(range(100), all_preds[:100], label="预测值", color="#9467bd", linewidth=2, marker="o", markersize=4)
ax3.plot(range(100), all_truths[:100], label="真实值", color="#8c564b", linewidth=2, marker="s", markersize=4)
ax3.fill_between(range(100), all_preds[:100], all_truths[:100], alpha=0.2, color="gray", label="误差区域")
ax3.set_title("GPU训练 - 前100个监测点预测对比", fontsize=14, fontweight="bold")
ax3.set_xlabel("监测点编号")
ax3.set_ylabel("交通流量值")
ax3.legend()
ax3.grid(alpha=0.3)

# 子图4：关键指标条形图
metrics = ["MAE", "RMSE", "MAPE(%)", "R²"]
values = [mae, rmse, mape, r2]
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor="black")
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
ax4.set_title("GPU训练 - 关键评估指标", fontsize=14, fontweight="bold")
ax4.set_ylabel("数值")
ax4.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "GPU_模型综合分析图表.png"), dpi=300, bbox_inches="tight")
plt.show()

# 清理GPU显存
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print("\n🗑️ 已清空GPU显存缓存")

# 最终提示
print(f"\n📁 所有GPU训练可视化图表已保存至: {os.path.abspath(SAVE_DIR)}")
print("\nGPU训练总结：")
print(f"1. 训练设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
print(f"2. 模型训练{EPOCHS}轮后收敛，最小测试损失: {min(test_loss_history):.4f}（归一化后）")
print(f"3. 预测效果：R²={r2:.4f}（越接近1越好），MAPE={mape:.2f}%（越小越好）")
print(f"4. 总训练耗时: {train_time_str}，GPU加速效果显著")