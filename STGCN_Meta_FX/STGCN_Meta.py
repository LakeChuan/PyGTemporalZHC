# @Time     : 2026
# @Author   : Veritas YIN (PyTorch版移植)
# @FileName : stgcn_pytorch.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : 移植自 https://github.com/VeritasYin/Project_Orion
# @Env      : PyTorch 2.8.0+cu128, NumPy 2.4.3

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse.linalg import eigs
from tqdm import tqdm

# ==================== 全局配置 ====================
# 配置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建输出文件夹
os.makedirs('./output/models', exist_ok=True)
os.makedirs('./output/tensorboard', exist_ok=True)
os.makedirs('./runresult', exist_ok=True)
# 日志保存路径
timestamp = time.strftime('%Y%m%d_%H%M%S')
train_log = f'./runresult/train_{timestamp}.txt'
test_log = f'./runresult/test_{timestamp}.txt'

def log_print(msg, file_path):
    """打印并保存日志，原版输出格式完全不变"""
    print(msg)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# ==================== 1. 图计算工具 (math_graph.py) ====================
def scaled_laplacian(W):
    """
    归一化图拉普拉斯矩阵计算，与原版完全一致
    :param W: 邻接矩阵 [n_route, n_route]
    :return: 缩放后的拉普拉斯矩阵
    """
    # 1. 计算节点度
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # 2. 初始化拉普拉斯矩阵 L = D - W
    L = -W
    L[np.diag_indices_from(L)] = d
    # 3. 对称归一化
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # 4. 缩放使最大特征值为2
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.asmatrix(2 * L / lambda_max - np.identity(n))  # 只改这行

def cheb_poly_approx(L, Ks, n):
    """切比雪夫多项式近似，原版逻辑不变"""
    L0, L1 = np.asmatrix(np.identity(n)), np.asmatrix(np.copy(L))  # 只改这行
    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.asmatrix(2 * L * L1 - L0)  # 只改这行
            L_list.append(np.copy(Ln))
            L0, L1 = np.asmatrix(np.copy(L1)), np.asmatrix(np.copy(Ln))  # 只改这行
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: 空间核大小必须大于1')

def first_approx(W, n):
    """1阶近似，原版保留"""
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    return np.mat(np.identity(n) + sinvD * A * sinvD)

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """加载权重矩阵，与原版完全一致"""
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        log_print(f'ERROR: 文件未找到 {file_path}', train_log)
        exit()

    if set(np.unique(W)) == {0, 1}:
        log_print('输入图为0/1矩阵，关闭scaling', train_log)
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W

# ==================== 2. 数学工具 (math_utils.py) ====================
def z_score(x, mean, std):
    """Z-score标准化"""
    return (x - mean) / std

def z_inverse(x, mean, std):
    """逆标准化"""
    return x * std + mean

def MAPE(v, v_):
    """平均绝对百分比误差"""
    return np.mean(np.abs(v_ - v) / (v + 1e-5))

def RMSE(v, v_):
    """均方根误差"""
    return np.sqrt(np.mean((v_ - v) ** 2))

def MAE(v, v_):
    """平均绝对误差"""
    return np.mean(np.abs(v_ - v))

def evaluation(y, y_, x_stats):
    """评估指标计算，多步/单步递归计算，原版逻辑不变"""
    dim = len(y_.shape)
    if dim == 3:
        # 单步预测
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # 多步预测
        tmp_list = []
        y = np.swapaxes(y, 0, 1)
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)

# ==================== 3. 数据加载 (data_utils.py 简化版) ====================
def gen_batch(data, batch_size, dynamic_batch=True, shuffle=True):
    """生成批次数据，完全还原原版逻辑"""
    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=not dynamic_batch)
    for batch in loader:
        yield batch[0].numpy()

class data_gen:
    """数据生成器，与原版结构、功能完全一致"""
    def __init__(self, file_path, n_split, n_route, n_frame):
        self.file_path = file_path
        self.n_train, self.n_val, self.n_test = n_split
        self.n_route = n_route
        self.n_frame = n_frame

        # 加载数据
        df = pd.read_csv(file_path, header=None)
        self.data = np.expand_dims(df.values, axis=-1)
        # 标准化
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = z_score(self.data, self.mean, self.std)
        # 生成序列
        self.seq = self._seq_gen()
        self._split_data()

    def _seq_gen(self):
        """生成时间序列"""
        seq = []
        for i in range(len(self.data) - self.n_frame + 1):
            seq.append(self.data[i:i + self.n_frame])
        return np.array(seq)

    def _split_data(self):
        """划分训练/验证/测试集"""
        n = len(self.seq)
        self.train = self.seq[:n - self.n_val - self.n_test]
        self.val = self.seq[n - self.n_val - self.n_test:n - self.n_test]
        self.test = self.seq[-self.n_test:]

    def get_data(self, category):
        """获取数据"""
        if category == 'train': return self.train
        elif category == 'val': return self.val
        elif category == 'test': return self.test

    def get_stats(self):
        """获取统计值"""
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, category):
        """获取数据长度"""
        return len(self.get_data(category))

# ==================== 4. 网络层 (layers.py) ====================
class GConv(nn.Module):
    """图卷积层，完全还原原版gconv逻辑"""
    def __init__(self, Ks, c_in, c_out, Lk):
        super(GConv, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        # 图核参数
        self.Lk = torch.FloatTensor(Lk).to(device)
        # 可学习参数theta
        self.theta = nn.Parameter(torch.FloatTensor(Ks * c_in, c_out))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x):
        B, T, N, C = x.shape
        # 维度变换：[B*T, N, C]
        x_reshape = x.reshape(-1, N, C)
        # 维度变换：[B*T, C, N]
        x_tmp = x_reshape.permute(0, 2, 1).contiguous()
        x_tmp = x_tmp.reshape(-1, N)
        # 图卷积计算
        x_mul = torch.matmul(x_tmp, self.Lk)
        x_mul = x_mul.reshape(-1, C, self.Ks, N)
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(-1, C * self.Ks)
        x_gconv = torch.matmul(x_ker, self.theta)
        return x_gconv.reshape(B, T, N, self.c_out)

class LayerNorm(nn.Module):
    """层归一化，还原原版layer_norm"""
    def __init__(self, n, c):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones([1, 1, n, c]))
        self.beta = nn.Parameter(torch.zeros([1, 1, n, c]))

    def forward(self, x):
        mu = x.mean(dim=[2,3], keepdim=True)
        sigma = x.var(dim=[2,3], keepdim=True) + 1e-6
        return (x - mu) / torch.sqrt(sigma) * self.gamma + self.beta

class TemporalConvLayer(nn.Module):
    """时间卷积层，原版逻辑1:1还原"""
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func

        # 残差连接
        if c_in != c_out:
            self.down_conv = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0)
        else:
            self.down_conv = None

        # 卷积核
        if act_func == 'GLU':
            self.conv = nn.Conv2d(c_in, 2*c_out, kernel_size=(Kt,1), padding=0)
            self.act = None
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=(Kt,1), padding=0)
            if act_func == 'relu':
                self.act = nn.ReLU()
            elif act_func == 'sigmoid':
                self.act = nn.Sigmoid()
            else:
                self.act = None

    def forward(self, x):
        B, T, N, C = x.shape
        # 维度转换 [B, C, T, N]
        x_in = x.permute(0,3,1,2).contiguous()
        residual = x_in

        # 卷积计算
        x_conv = self.conv(x_in)

        # 残差处理
        if self.down_conv is not None:
            residual = self.down_conv(residual)
        residual = residual[:, :, self.Kt-1:T, :]

        # 激活函数
        if self.act_func == 'GLU':
            out = x_conv[:, :self.c_out, :, :]
            gate = torch.sigmoid(x_conv[:, self.c_out:, :, :])
            x_out = (out + residual) * gate
        else:
            x_out = x_conv
            if self.act is not None:
                x_out = self.act(x_out + residual)

        # 维度转换回 [B, T, N, C]
        return x_out.permute(0,2,3,1).contiguous()

class SpatialConvLayer(nn.Module):
    """空间图卷积层，原版逻辑不变"""
    def __init__(self, Ks, c_in, c_out, Lk):
        super(SpatialConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gconv = GConv(Ks, c_in, c_out, Lk)

        if c_in != c_out:
            self.down_conv = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0)
        else:
            self.down_conv = None

    def forward(self, x):
        B, T, N, C = x.shape
        # 残差
        if self.down_conv is not None:
            x_in = x.permute(0,3,1,2).contiguous()
            x_in = self.down_conv(x_in).permute(0,2,3,1).contiguous()
        else:
            x_in = x
        # 图卷积
        x_gc = self.gconv(x)
        return torch.relu(x_gc + x_in)

class STConvBlock(nn.Module):
    """时空卷积块，与原版结构完全一致"""
    def __init__(self, Ks, Kt, channels, Lk, keep_prob=0.0):
        super(STConvBlock, self).__init__()
        c_si, c_t, c_oo = channels
        # 时间卷积 -> 空间卷积 -> 时间卷积
        self.temp1 = TemporalConvLayer(Kt, c_si, c_t, act_func='GLU')
        self.spatial = SpatialConvLayer(Ks, c_t, c_t, Lk)
        self.temp2 = TemporalConvLayer(Kt, c_t, c_oo, act_func='GLU')
        self.norm = LayerNorm(228, c_oo)
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, x):
        x = self.temp1(x)
        x = self.spatial(x)
        x = self.temp2(x)
        x = self.norm(x)
        return self.dropout(x)

class OutputLayer(nn.Module):
    """输出层，原版逻辑不变"""
    def __init__(self, T, channel, n_route):
        super(OutputLayer, self).__init__()
        self.temp1 = TemporalConvLayer(T, channel, channel, act_func='GLU')
        self.norm = LayerNorm(n_route, channel)
        self.temp2 = TemporalConvLayer(1, channel, channel, act_func='sigmoid')
        self.fc = nn.Conv2d(channel, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.temp1(x)
        x = self.norm(x)
        x = self.temp2(x)
        # 全连接输出
        x = x.permute(0,3,1,2).contiguous()
        x = self.fc(x)
        return x.permute(0,2,3,1).contiguous()

# ==================== 5. 模型主体 (base_model.py) ====================
class STGCN(nn.Module):
    """STGCN主模型，完全还原原版build_model"""
    def __init__(self, n_his, Ks, Kt, blocks, Lk, n_route=228):
        super(STGCN, self).__init__()
        self.n_his = n_his
        self.blocks = nn.ModuleList()
        Ko = n_his

        # 堆叠时空卷积块
        for channels in blocks:
            self.blocks.append(STConvBlock(Ks, Kt, channels, Lk))
            Ko -= 2 * (Kt - 1)

        # 输出层
        if Ko > 1:
            self.output = OutputLayer(Ko, blocks[-1][-1], n_route)
        else:
            raise ValueError('输出层卷积核大小必须大于1')

    def forward(self, x):
        # 输入：[batch, his+1, n, 1]，取前n_his步
        x = x[:, :self.n_his, :, :]
        # 时空块前向传播
        for block in self.blocks:
            x = block(x)
        # 输出层
        y = self.output(x)
        return y

# ==================== 6. 训练/测试逻辑 (trainer.py + tester.py) ====================
def multi_pred(model, seq, batch_size, n_his, n_pred, step_idx):
    """多步预测，与原版multi_pred逻辑完全一致"""
    pred_list = []
    model.eval()
    with torch.no_grad():
        for i in gen_batch(seq, min(batch_size, len(seq))):
            test_seq = np.copy(i)
            step_list = []
            for j in range(n_pred):
                # 转换为tensor
                x = torch.FloatTensor(test_seq).to(device)
                pred = model(x)
                pred = pred.squeeze(1).cpu().numpy()
                # 更新序列
                test_seq[:, 0:n_his-1, :, :] = test_seq[:, 1:n_his, :, :]
                test_seq[:, n_his-1, :, :] = pred
                step_list.append(pred)
            pred_list.append(step_list)

    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]

def model_inference(model, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    """模型推理，原版逻辑不变"""
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    # 验证集评估
    y_val, len_val = multi_pred(model, x_val, batch_size, n_his, n_pred, step_idx)
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)

    # 更新最优指标
    chks = evl_val < min_va_val
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(model, x_test, batch_size, n_his, n_pred, step_idx)
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        min_val = evl_pred
    return min_va_val, min_val

def model_train(inputs, blocks, args):
    """模型训练，完全还原原版训练逻辑、学习率、优化器"""
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, opt = args.batch_size, args.epoch, args.opt

    # 初始化模型
    model = STGCN(n_his, Ks, Kt, blocks, Lk, n).to(device)
    log_print(f'模型参数量: {sum(p.numel() for p in model.parameters())}', train_log)

    # 优化器与学习率衰减（与原版一致：每5轮衰减0.7）
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr) if opt == 'RMSProp' else optim.Adam(model.parameters(), lr=args.lr)
    epoch_step = inputs.get_len('train') // batch_size + 1
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    # 推理模式配置
    if args.inf_mode == 'sep':
        step_idx = n_pred - 1
        tmp_idx = [step_idx]
        min_val = min_va_val = np.array([4e1, 1e5, 1e5])
    else:
        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))

    # 训练循环
    for i in range(epoch):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        # 批次训练
        for j, x_batch in enumerate(gen_batch(inputs.get_data('train'), batch_size, shuffle=True)):
            x = torch.FloatTensor(x_batch).to(device)
            optimizer.zero_grad()
            # 前向传播
            pred = model(x)
            # 损失函数（与原版一致：L2损失 + 复制损失）
            loss = torch.nn.functional.mse_loss(pred, x[:, n_his:n_his+1, :, :])
            copy_loss = torch.nn.functional.mse_loss(x[:, n_his-1:n_his, :, :], x[:, n_his:n_his+1, :, :])
            total_loss = loss + copy_loss
            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 打印中间损失
            if j % 50 == 0:
                log_print(f'Epoch {i:2d}, Step {j:3d}: [{loss:.3f}, {copy_loss:.3f}]', train_log)

        # 学习率衰减
        if (i+1) % 5 == 0:
            scheduler.step()

        log_print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s', train_log)

        # 推理评估
        start_time = time.time()
        min_va_val, min_val = model_inference(model, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)
        # 打印评估结果
        for ix in tmp_idx:
            va, te = min_va_val[ix-2:ix+1], min_val[ix-2:ix+1]
            log_print(f'Time Step {ix + 1}: MAPE {va[0]:7.3%}, {te[0]:7.3%}; MAE  {va[1]:4.3f}, {te[1]:4.3f}; RMSE {va[2]:6.3f}, {te[2]:6.3f}.', train_log)
        log_print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s', train_log)

        # 保存模型
        if (i + 1) % args.save == 0:
            torch.save(model.state_dict(), f'./output/models/STGCN_epoch_{i+1}.pth')
            log_print(f'<< 模型已保存至 ./output/models/STGCN_epoch_{i+1}.pth', train_log)

    log_print('Training model finished!', train_log)
    return model

def model_test(inputs, batch_size, n_his, n_pred, inf_mode):
    """模型测试，还原原版测试逻辑"""
    log_print("\n" + "="*50, test_log)
    log_print(f'测试时间: {time.strftime("%Y-%m-%d %H:%M:%S")}', test_log)
    # 加载最优模型
    model = STGCN(n_his, Ks, Kt, blocks, Lk).to(device)
    model.load_state_dict(torch.load(f'./output/models/STGCN_epoch_{args.epoch}.pth'))
    log_print('>> 加载训练完成的模型', test_log)

    # 测试模式
    if inf_mode == 'sep':
        step_idx = n_pred - 1
        tmp_idx = [step_idx]
    else:
        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1

    # 测试推理
    x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
    y_test, len_test = multi_pred(model, x_test, batch_size, n_his, n_pred, step_idx)
    evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)

    # 打印结果
    for ix in tmp_idx:
        te = evl[ix - 2:ix + 1]
        log_print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.', test_log)
    log_print("="*50, test_log)
    log_print('Testing model finished!', test_log)

# ==================== 主函数 (main.py) ====================
if __name__ == '__main__':
    # 命令行参数（与原版完全一致）
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_route', type=int, default=228)
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--ks', type=int, default=3)
    parser.add_argument('--kt', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, default='RMSProp')
    parser.add_argument('--graph', type=str, default='default')
    parser.add_argument('--inf_mode', type=str, default='merge')
    args = parser.parse_args()
    log_print(f'Training configs: {args}', train_log)

    # 全局参数
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    blocks = [[1, 32, 64], [64, 32, 128]]

    # 1. 加载图矩阵
    log_print("加载图邻接矩阵...", train_log)
    if args.graph == 'default':
        W = weight_matrix(f'./dataset/PeMSD7_W_{n}.csv')
    else:
        W = weight_matrix(f'./dataset/{args.graph}')

    # 2. 计算图核
    L = scaled_laplacian(W)
    Lk = cheb_poly_approx(L, Ks, n)

    # 3. 数据预处理
    log_print("加载交通数据...", train_log)
    data_file = f'PeMSD7_V_{n}.csv'
    PeMS = data_gen(f'./dataset/{data_file}', (34, 5, 5), n, n_his + n_pred)
    log_print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}', train_log)

    # 4. 训练 + 测试
    trained_model = model_train(PeMS, blocks, args)
    model_test(PeMS, args.batch_size, n_his, n_pred, args.inf_mode)

    log_print("\n✅ 全部运行完成！结果保存在 ./runresult/", train_log)