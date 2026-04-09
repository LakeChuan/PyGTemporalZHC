import torch
import torch.nn as nn
import numpy as np

# ======================== 1. 定义基础层 ========================
class Align(nn.Module):
    """通道对齐层：统一输入输出通道数（1x1卷积或恒等映射）"""
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=False) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.conv(x)


class CausalConv2d(nn.Module):
    """因果卷积层：保证只使用当前/过去时间步信息"""
    def __init__(self, in_channels, out_channels, kernel_size, enable_padding=False, dilation=1):
        super(CausalConv2d, self).__init__()
        self.enable_padding = enable_padding
        # 计算padding：仅在时间维度padding，节点维度不padding
        pad_left = dilation * (kernel_size[0] - 1) if enable_padding else 0
        self.pad = (pad_left, 0)
        # 定义卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(pad_left, 0),  # (time_pad, vertex_pad)
            dilation=dilation,
            bias=False
        )
        # 初始化权重（固定值，方便计算）
        nn.init.constant_(self.conv.weight, 1.0)

    def forward(self, x):
        x = self.conv(x)
        # 裁剪padding以保证因果性（去掉未来时间步）
        if self.enable_padding and self.pad[0] > 0:
            x = x[:, :, :-self.pad[0], :]
        return x


class TemporalConvLayer(nn.Module):
    """门控时间卷积层（GLU/GTU）"""
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.act_func = act_func

        # 定义因果卷积
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1
            )

        self.relu = nn.ReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        # 残差连接对齐
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        # 因果卷积
        x_causal_conv = self.causal_conv(x)

        # 激活函数分支
        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, :self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'激活函数 {self.act_func} 未实现')

        return x


class GraphConvLayer(nn.Module):
    """图卷积层（支持普通图卷积/切比雪夫图卷积）"""
    def __init__(self, conv_type, c_in, c_out, Ks, gso, bias=False):
        super(GraphConvLayer, self).__init__()
        self.conv_type = conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso  # 图移位算子（邻接矩阵）
        self.bias = bias

        # 初始化权重（固定值，方便计算）
        self.weight = nn.Parameter(torch.eye(c_in, c_out).unsqueeze(0), requires_grad=False)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(c_out), requires_grad=False)
        else:
            self.bias_param = None

    def forward(self, x):
        bs, c_in, ts, n_vertex = x.shape

        # 维度调整：[bs, c_in, ts, n_vertex] → [bs*ts, n_vertex, c_in]
        x_reshaped = x.permute(0, 2, 3, 1).reshape(bs * ts, n_vertex, c_in)

        # 图卷积计算
        if self.conv_type == 'ChebGraphConv':
            # 切比雪夫多项式展开（简化版，Ks=1）
            x_list = [x_reshaped]
            for k in range(1, self.Ks):
                x_k = torch.matmul(self.gso, x_list[k-1])
                x_list.append(x_k)
            x_conv = sum([torch.matmul(x_k, self.weight[k]) for k in range(self.Ks)])
        else:
            # 普通图卷积：x' = GSO * x * weight
            x_conv = torch.matmul(torch.matmul(self.gso, x_reshaped), self.weight[0])

        # 加偏置
        if self.bias_param is not None:
            x_conv = x_conv + self.bias_param

        # 恢复维度：[bs*ts, n_vertex, c_out] → [bs, c_out, ts, n_vertex]
        x_conv = x_conv.reshape(bs, ts, n_vertex, self.c_out).permute(0, 3, 1, 2)

        return x_conv


class STConvBlock(nn.Module):
    """时空卷积块（TGTND结构）"""
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        # 时间卷积1
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        # 图卷积
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        # 时间卷积2
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        # 层归一化
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        # 激活和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        # 步骤1：时间卷积1（提取时间特征）
        x1 = self.tmp_conv1(x)
        # 步骤2：图卷积（提取空间特征）+ ReLU
        x2 = self.graph_conv(x1)
        x2 = self.relu(x2)
        # 步骤3：时间卷积2（二次提取时空特征）
        x3 = self.tmp_conv2(x2)
        # 步骤4：层归一化
        x4 = self.tc2_ln(x3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 步骤5：Dropout
        x5 = self.dropout(x4)

        # 返回各步骤结果，方便分析
        return {
            'tmp_conv1': x1,
            'graph_conv': x2,
            'tmp_conv2': x3,
            'layer_norm': x4,
            'dropout': x5
        }

# ======================== 2. 测试代码 ========================
if __name__ == "__main__":
    # 固定随机种子（保证结果可复现）
    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------- 2.1 定义测试参数 --------------------
    bs = 1               # 批量大小
    c_in = 2             # 输入通道数
    ts = 5               # 时间步长度
    n_vertex = 3         # 节点数
    Kt = 2               # 时间卷积核大小
    Ks = 1               # 图卷积阶数
    channels = [2, 2, 2] # 通道数配置
    act_func = 'glu'     # 激活函数
    graph_conv_type = 'GraphConv'  # 图卷积类型
    bias = False         # 图卷积是否加偏置
    droprate = 0.0       # Dropout概率（0=不丢弃）

    # -------------------- 2.2 构造测试数据 --------------------
    # 输入张量：[bs, c_in, ts, n_vertex]
    x = torch.tensor([
        # c_in=0 通道
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        # c_in=1 通道
        [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]]
    ], dtype=torch.float32).unsqueeze(0)

    # 图移位算子（邻接矩阵）：3节点的简单连接
    gso = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=torch.float32)

    # -------------------- 2.3 初始化模型并运行 --------------------
    st_block = STConvBlock(
        Kt=Kt, Ks=Ks, n_vertex=n_vertex,
        last_block_channel=c_in, channels=channels,
        act_func=act_func, graph_conv_type=graph_conv_type,
        gso=gso, bias=bias, droprate=droprate
    )

    # 前向传播
    results = st_block(x)

    # -------------------- 2.4 输出结果分析 --------------------
    print("=" * 60)
    print("输入张量信息：")
    print(f"输入shape: {x.shape}")
    # 修复点1：输入张量无梯度，可直接转numpy
    print(f"输入值:\n {x.squeeze(0).numpy()}\n")

    # 遍历输出各步骤结果
    for step_name, tensor in results.items():
        print("=" * 60)
        print(f"{step_name} 结果：")
        print(f"shape: {tensor.shape}")
        # 修复点2：有梯度的张量先detach()再转numpy
        print(f"值:\n {tensor.detach().squeeze(0).numpy()}")
        print("-" * 60)