import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================= 基础组件: 图卷积 (ChebNet) ================= #
class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, cheb_polys):
        super(ChebConv, self).__init__()
        self.K = K
        self.cheb_polys = cheb_polys
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_c, out_c))
        nn.init.xavier_uniform_(self.Theta)
        self.bias = nn.Parameter(torch.FloatTensor(out_c))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, N, C_in)
        batch_size, num_nodes, in_channels = x.shape
        outputs = []
        for k in range(self.K):
            T_k = self.cheb_polys[k].to(x.device)
            support = torch.einsum("nm,bmc->bnc", T_k, x)
            outputs.append(support)
        outputs = torch.stack(outputs, dim=0)
        y = torch.einsum("kbnc,kcd->bnd", outputs, self.Theta)
        return F.relu(y + self.bias)


# ================= 核心组件: STGCN Block (三明治结构) ================= #
class STGCN_Block(nn.Module):
    def __init__(self, in_c, out_c, K, cheb_polys):
        super(STGCN_Block, self).__init__()
        # 1. Temporal Conv 1
        self.time_conv1 = nn.Conv2d(in_c, 2 * out_c, kernel_size=(1, 3), padding=(0, 1))
        # 2. Spatial Graph Conv
        self.cheb_conv = ChebConv(out_c, out_c, K, cheb_polys)
        # 3. Temporal Conv 2
        self.time_conv2 = nn.Conv2d(out_c, 2 * out_c, kernel_size=(1, 3), padding=(0, 1))
        # 4. Residual
        self.residual_conv = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(out_c)

    def forward(self, x):
        # x: (B, C, N, T)
        # Step 1: Temporal
        x_t1 = self.time_conv1(x)
        P, Q = x_t1.chunk(2, dim=1)
        x_t1 = P * torch.sigmoid(Q)

        # Step 2: Spatial
        B, C, N, T = x_t1.shape
        x_g_in = x_t1.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x_g = self.cheb_conv(x_g_in)
        x_g = x_g.reshape(B, T, N, C).permute(0, 3, 2, 1)

        # Step 3: Temporal
        x_t2 = self.time_conv2(x_g)
        P2, Q2 = x_t2.chunk(2, dim=1)
        x_t2 = P2 * torch.sigmoid(Q2)

        # Step 4: Residual
        x_res = self.residual_conv(x)
        return self.ln((x_t2 + x_res).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ================= 模型 1: Baseline LSTM ================= #
class Baseline_LSTM(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_len):
        super(Baseline_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.lstm = nn.LSTM(input_size=num_nodes, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_nodes * output_len)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.squeeze(-1)
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred.reshape(B, self.output_len, N, 1)


# ================= 模型 2: FC-LSTM ================= #
class FC_LSTM(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_len):
        super(FC_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.lstm = nn.LSTM(input_size=num_nodes * input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_nodes * output_len)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat = x.reshape(B, T, N * C)
        out, _ = self.lstm(x_flat)
        pred = self.fc(out[:, -1, :])
        return pred.reshape(B, self.output_len, N, 1)


# ================= 模型 3: DCRNN ================= #
class DCRNN_Cell(nn.Module):
    def __init__(self, num_units, adj_mx):
        super().__init__()
        self.fc_x = nn.Linear(1, num_units)
        self.fc_h = nn.Linear(num_units, num_units)
        self.adj = torch.tensor(adj_mx).float() if adj_mx is not None else None

    def forward(self, x, h):
        if self.adj is not None:
            if x.device != self.adj.device: self.adj = self.adj.to(x.device)
            x = torch.einsum('nm,bmc->bnc', self.adj, x)
        gates = torch.sigmoid(self.fc_x(x) + self.fc_h(h))
        return gates * h + (1 - gates) * torch.tanh(self.fc_x(x))


class DCRNN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_len, adj_mx):
        super(DCRNN, self).__init__()
        self.cell = DCRNN_Cell(hidden_dim, adj_mx)
        self.fc = nn.Linear(hidden_dim, output_len)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, T, N, C = x.shape
        h = torch.zeros(B, N, self.hidden_dim).to(x.device)
        for t in range(T):
            h = self.cell(x[:, t, :, :], h)
        pred = self.fc(h)
        return pred.unsqueeze(-1).permute(0, 2, 1, 3)


# ================= 模型 4: STGCN ================= #
class STGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_len, K, cheb_polys):
        super(STGCN, self).__init__()
        self.block1 = STGCN_Block(input_dim, hidden_dim, K, cheb_polys)
        self.block2 = STGCN_Block(hidden_dim, hidden_dim, K, cheb_polys)
        self.final_conv = nn.Conv2d(hidden_dim, output_len, kernel_size=(1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = x[:, :, :, -1:]
        x = self.final_conv(x)
        return x.permute(0, 1, 2, 3)


# ================= 模型 5: APST-Net (最优并联模型) ================= #
class APST_Net(nn.Module):
    def __init__(self, model_a, model_b):
        super(APST_Net, self).__init__()
        self.model_a = model_a  # FC-LSTM (Temporal)
        self.model_b = model_b  # STGCN (Spatial)

        # 动态门控网络: 输入(B, T, N, 2) -> 输出(B, T, N, 1)
        self.gate_fc = nn.Linear(2, 1)

    def forward(self, x):
        # 1. 获取两个分支的输出
        out_temporal = self.model_a(x)  # FC-LSTM
        out_spatial = self.model_b(x)  # STGCN

        # 2. 拼接 (Concat) -> (Batch, T, N, 2)
        stacked = torch.cat([out_spatial, out_temporal], dim=-1)

        # 3. 门控计算
        z = torch.sigmoid(self.gate_fc(stacked))  # z 是 Spatial 的权重

        # 4. 加权融合
        fused_out = z * out_spatial + (1 - z) * out_temporal

        # 返回 预测结果 和 空间权重z
        return fused_out, z


class APST_FixedGate(nn.Module):
    def __init__(self, model_a, model_b, spatial_weight=0.5):
        super(APST_FixedGate, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.spatial_weight = float(spatial_weight)

    def forward(self, x):
        out_temporal = self.model_a(x)
        out_spatial = self.model_b(x)
        z = torch.full_like(out_spatial, self.spatial_weight)
        fused_out = z * out_spatial + (1 - z) * out_temporal
        return fused_out, z


class APST_GlobalGate(nn.Module):
    def __init__(self, model_a, model_b, init_spatial_weight=0.5):
        super(APST_GlobalGate, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        init_spatial_weight = min(max(init_spatial_weight, 1e-4), 1 - 1e-4)
        init_logit = math.log(init_spatial_weight / (1 - init_spatial_weight))
        self.gate_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, x):
        out_temporal = self.model_a(x)
        out_spatial = self.model_b(x)
        z_scalar = torch.sigmoid(self.gate_logit)
        z = torch.ones_like(out_spatial) * z_scalar
        fused_out = z * out_spatial + (1 - z) * out_temporal
        return fused_out, z


class APST_TemporalThenSpatial(nn.Module):
    def __init__(self, temporal_model, spatial_model):
        super(APST_TemporalThenSpatial, self).__init__()
        self.temporal_model = temporal_model
        self.spatial_model = spatial_model

    def forward(self, x):
        temporal_out = self.temporal_model(x)
        refined_out = self.spatial_model(temporal_out)
        return refined_out