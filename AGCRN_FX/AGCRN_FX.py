import os
import sys
import math
import time
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import configparser
from datetime import datetime
import matplotlib.pyplot as plt

# ===================== 1. 路径适配（和你的目录完全对应）=====================
# 当前文件所在目录（AGCRN_FX）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集目录（../DATA）
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "DATA")

# 全局配置
Mode = 'Train'
DEBUG = True
DATASET = 'PEMSD4'  # PEMSD4 or PEMSD8
DEVICE = 'cuda:0'
MODEL = 'AGCRN'

# 读取同目录下conf文件
config_file = os.path.join(BASE_DIR, f'{DATASET}_{MODEL}.conf')
print(f'Reading config file: {config_file}')
config = configparser.ConfigParser()
config.read(config_file)


# ===================== 2. 工具函数 =====================
def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print(f'Total params num: {total_num}')
    print('*****************Finish Parameter****************')


# 日志工具（已增强：自动保存 log.txt）
def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 自动保存 log.txt
    log_path = os.path.join(root, "log.txt")
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# 评价指标
def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def RMSE_torch(pred, true, mask_value=None):
    return torch.sqrt(MSE_torch(pred, true, mask_value))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RRSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR_torch(pred, true, mask_value=None):
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(1).unsqueeze(1)
        true = true.unsqueeze(1).unsqueeze(1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(1)
        true = true.transpose(1, 2).unsqueeze(1)
    elif len(pred.shape) == 4:
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    dims = (0, 1, 2)
    pred_mean = pred.mean(dims)
    true_mean = true.mean(dims)
    pred_std = pred.std(dims)
    true_std = true.std(dims)
    corr = ((pred - pred_mean) * (true - true_mean)).mean(dims) / (pred_std * true_std)
    idx = (true_std != 0)
    return corr[idx].mean()


def All_Metrics(pred, true, mask1, mask2):
    if isinstance(pred, np.ndarray):
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mape = np.mean(np.abs((true - pred) / true))
        rrse = np.sqrt(np.sum((pred - true) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
        corr = 0
    elif isinstance(pred, torch.Tensor):
        mae = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
    else:
        raise TypeError
    return mae, rmse, mape, rrse, corr


# 标准化器
class NScaler(object):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.mean, np.ndarray):
            self.mean = torch.from_numpy(self.mean).to(data.device)
            self.std = torch.from_numpy(self.std).to(data.device)
        return data * self.std + self.mean


class MinMax01Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device)
            self.max = torch.from_numpy(self.max).to(data.device)
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2 - 1

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device)
            self.max = torch.from_numpy(self.max).to(data.device)
        return ((data + 1) / 2) * (self.max - self.min) + self.min


# 数据滑窗
def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    length = len(data)
    end = length - horizon - window + 1
    X, Y = [], []
    idx = 0
    if single:
        while idx < end:
            X.append(data[idx:idx + window])
            Y.append(data[idx + window + horizon - 1:idx + window + horizon])
            idx += 1
    else:
        while idx < end:
            X.append(data[idx:idx + window])
            Y.append(data[idx + window:idx + window + horizon])
            idx += 1
    return np.array(X), np.array(Y)


# 加载数据集（适配你的DATA目录）
def load_st_dataset(dataset):
    if dataset == 'PEMSD4':
        data_path = os.path.join(DATA_DIR, 'PEMS04', 'pems04.npz')
    elif dataset == 'PEMSD8':
        data_path = os.path.join(DATA_DIR, 'PEMS08', 'pems08.npz')
    else:
        raise ValueError
    data = np.load(data_path)['data'][:, :, 0]
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print(f'Load {dataset} Dataset shaped: {data.shape}')
    return data


# 数据加载器
def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        mn = data.min(axis=0, keepdims=True) if column_wise else data.min()
        mx = data.max(axis=0, keepdims=True) if column_wise else data.max()
        scaler = MinMax01Scaler(mn, mx)
    elif normalizer == 'max11':
        mn = data.min(axis=0, keepdims=True) if column_wise else data.min()
        mx = data.max(axis=0, keepdims=True) if column_wise else data.max()
        scaler = MinMax11Scaler(mn, mx)
    elif normalizer == 'std':
        mean = data.mean(axis=0, keepdims=True) if column_wise else data.mean()
        std = data.std(axis=0, keepdims=True) if column_wise else data.std()
        scaler = StandardScaler(mean, std)
    elif normalizer == 'None':
        scaler = NScaler()
    else:
        raise ValueError
    return scaler.transform(data), scaler


def split_data_by_ratio(data, val_ratio, test_ratio):
    L = data.shape[0]
    test = data[-int(L * test_ratio):]
    val = data[-int(L * (val_ratio + test_ratio)):-int(L * test_ratio)]
    train = data[:-int(L * (val_ratio + test_ratio))]
    return train, val, test


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    ds = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_dataloader(args, normalizer='std', tod=False, dow=False, weather=False, single=True):
    data = load_st_dataset(args.dataset)
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    train, val, test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    x_tr, y_tr = Add_Window_Horizon(train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(val, args.lag, args.horizon, single)
    x_te, y_te = Add_Window_Horizon(test, args.lag, args.horizon, single)
    print('Train:', x_tr.shape, y_tr.shape)
    print('Val:', x_val.shape, y_val.shape)
    print('Test:', x_te.shape, y_te.shape)
    train_dl = data_loader(x_tr, y_tr, args.batch_size, True, True)
    val_dl = data_loader(x_val, y_val, args.batch_size, False, True) if len(x_val) else None
    test_dl = data_loader(x_te, y_te, args.batch_size, False, False)
    return train_dl, val_dl, test_dl, scaler


# ===================== 3. 模型定义 =====================
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super().__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class AGCRN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim,
                                args.num_layers)
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)
        return output


# ===================== 4. 训练器（只新增功能）=====================
class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(f'Experiment log path in: {args.log_dir}')

        # ========== 新增：早停 + 绘图记录 ==========
        self.train_loss_list = []
        self.val_loss_list = []

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info(f'**********Val Epoch {epoch}: average Loss: {val_loss:.6f}')
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            self.optimizer.zero_grad()
            teacher_forcing_ratio = 1. if not self.args.teacher_forcing else self._compute_sampling_threshold(
                (epoch - 1) * self.train_per_epoch + batch_idx, self.args.tf_decay_steps)
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output, label)
            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % self.args.log_step == 0:
                self.logger.info(f'Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.6f}')
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            f'**********Train Epoch {epoch}: averaged Loss: {train_epoch_loss:.6f}, tf_ratio: {teacher_forcing_ratio:.6f}')
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_dataloader = self.val_loader if self.val_loader is not None else self.test_loader
            val_loss = self.val_epoch(epoch, val_dataloader)

            # ========== 新增：保存loss曲线 ==========
            self.train_loss_list.append(train_loss)
            self.val_loss_list.append(val_loss)

            if train_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if val_loss < best_loss:
                best_loss = val_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            # ========== 新增：早停触发 ==========
            if self.args.early_stop and not_improved_count >= self.args.early_stop_patience:
                self.logger.info(f"Early stop triggered! Patience={self.args.early_stop_patience}")
                break

            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        # ========== 新增：绘制loss曲线并保存 ==========
        plt.figure()
        plt.plot(self.train_loss_list, label='Train Loss')
        plt.plot(self.val_loss_list, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig(os.path.join(self.args.log_dir, 'loss_curve.png'))
        plt.close()

        training_time = time.time() - start_time
        self.logger.info(f"Total training time: {training_time / 60:.4f}min, best loss: {best_loss:.6f}")
        torch.save(best_model, self.best_path)
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        os.makedirs(args.log_dir, exist_ok=True)
        np.save(os.path.join(args.log_dir, f'{args.dataset}_true.npy'), y_true.cpu().numpy())
        np.save(os.path.join(args.log_dir, f'{args.dataset}_pred.npy'), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
            logger.info(f"Horizon {t + 1:02d}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info(f"Average Horizon, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))


# ===================== 5. 主函数 =====================
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss


def main():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--mode', default=Mode, type=str)
    parser.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
    parser.add_argument('--debug', default=DEBUG, type=eval)
    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--cuda', default=True, type=bool)
    # data
    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    # model
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # train
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser.add_argument('--teacher_forcing', default=False, type=bool)
    parser.add_argument('--tf_decay_steps', default=2000, type=int)
    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval)
    # test
    parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)
    args = parser.parse_args()

    init_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'

    # init model
    model = AGCRN(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    # load dataset
    train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                                   normalizer=args.normalizer,
                                                                   tod=args.tod, dow=False,
                                                                   weather=False, single=False)

    # init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        print(f"Warning: loss_func={args.loss_func} not recognized, using MSELoss by default")
        loss = torch.nn.MSELoss().to(args.device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                 weight_decay=0, amsgrad=False)

    # learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)

    # ========== 新增：数据集_时间戳 文件夹 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_{timestamp}"
    args.log_dir = os.path.join(BASE_DIR, "results", exp_name)
    os.makedirs(args.log_dir, exist_ok=True)

    # start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=lr_scheduler)
    if args.mode.lower() == 'train':
        trainer.train()
    elif args.mode.lower() == 'test':
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'pre-trained', f'{args.dataset}.pth')))
        print("Load saved model")
        trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
    else:
        trainer.train()


if __name__ == "__main__":
    main()