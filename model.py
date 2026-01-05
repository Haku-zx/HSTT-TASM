from time import sleep
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GatedLinearUnit(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)


        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb


class TemporalAggregationConv(nn.Module):
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(TemporalAggregationConv, self).__init__()
        layers = []
        kernel_size = int(length / layer + 1)
        for i in range(layer):
            self.conv = nn.Conv2d(features, features, (1, kernel_size))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.tcn(x) + x[..., -1].unsqueeze(-1)
        return x


class SpatialMemoryAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        super(SpatialMemoryAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

        self.memory = nn.Parameter(torch.randn(head, seq_length, num_nodes, self.d_k))
        nn.init.xavier_uniform_(self.memory)

        self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
        ]

    def forward(self, input, adj_list=None):
        query, value = self.q(input), self.v(input)

        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(
            0, 1, 4, 3, 2
        )

        key = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        Aapt = torch.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1
        )
        kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)

        attn_dyn = torch.einsum("nm,bhlnc->bhlnc", Aapt, value)

        x = attn_qkv + attn_dyn
        x = (
            x.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(x.shape[0], self.d_model, self.num_nodes, self.seq_length)
        )
        x = self.concat(x)
        if self.num_nodes not in [170, 358, 5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias

class SpatialPerceptionEncoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(SpatialPerceptionEncoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialMemoryAttention(
            device, d_model, head, num_nodes, seq_length=seq_length
        )
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GatedLinearUnit(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input, adj_list=None):
        # 64 64 170 12
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x


class DualChannelLearner(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(DualChannelLearner, self).__init__()

        self.low_freq_layers = nn.ModuleList([
            TemporalAttention(features, num_nodes, length) for _ in range(layers)
        ])

        kernel_size = int(length / layers + 1)


        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, XL, XH):
        res_xl = XL
        res_xh = XH

        for layer in self.low_freq_layers:
            XL = layer(XL)

        XL = (res_xl[..., -1] + XL[..., -1]).unsqueeze(-1)

        XH = nn.functional.pad(XH, (1, 0, 0, 0))

        for layer in self.high_freq_layers:
            XH = layer(XH)

        XH = (res_xh[..., -1] + XH[..., -1]).unsqueeze(-1)

        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * XL + (1 - alpha_sigmoid) * XH

        return output


class TemporalAttention(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TemporalAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.conv2 = nn.Conv2d(
            num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        # nn.init.xavier_uniform_(self.b)
        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):

        seq = seq.transpose(3, 2)

        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()

        logits = self.bn(logits).permute(0, 2, 1).contiguous()

        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)

        x_1 = torch.einsum('bcnl,blq->bcnq', seq, T_coef)

        return x_1


class HSTT_TASM(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358 or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes > 200:
            time = 96

        # TE
        self.TE = TemporalEmbedding(time, channels)


        self.TAC = TemporalAggregationConv(channels, layer=4, length=self.input_len)

        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))

        self.network_channel = channels * 2

        # SPE
        self.SPE = SpatialPerceptionEncoder(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )

        self.start_conv_1 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.start_conv_2 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))

        self.layers = 2
        self.dims = 6


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data

        history_data = history_data.permute(0, 3, 2, 1)

        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()
        coef = pywt.wavedec(residual_numpy, 'db1', level=2)
        coefl = [coef[0]] + [None] * (len(coef) - 1)
        coefh = [None] + coef[1:]

        input_data_1 = self.start_conv_1(xl)
        input_data_2 = self.start_conv_2(xh)
        input_data = self.DCL(input_data_1, input_data_2)



        data_st = self.SPE(data_st) + self.fc_st(data_st)

        prediction = self.regression_layer(data_st)

        return prediction
