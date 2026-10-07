import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils1.func import get_scale_table
from model.compression_modules import *
import math
from compressai.ops import LowerBound
from compressai.models import CompressionModel

class EncoderPredictor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid() 
        )

    def forward(self, y):
        return self.net(y)

class STanHQuantizer(nn.Module):
    def __init__(self, channels, num_sigmoids=64, extrema=10, symmetry=True):
        super().__init__()
        self.num_sigmoids = int(num_sigmoids)
        self.symmetry = symmetry
        self.extrema = extrema
        self.jump = float(extrema) / num_sigmoids if num_sigmoids > 0 else 1.0
        
        # 🟢 恢复为全局参数 (1D)，不再区分通道
        if self.symmetry:
            self.w = nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump)
            initial_b = torch.arange(self.jump/2, extrema + self.jump/2, self.jump)
            self.b = nn.Parameter(initial_b[:self.num_sigmoids])
        else:
            self.w = nn.Parameter(torch.zeros(self.num_sigmoids * 2) + self.jump)
            self.b = nn.Parameter(torch.arange(-extrema + 0.5, extrema + 0.5, self.jump))

        self.register_buffer("cum_w", torch.Tensor())
        self.register_buffer("average_points", torch.Tensor())
        self.register_buffer("distance_points", torch.Tensor())

    def _get_params(self):
        if self.symmetry:
            sym_w = torch.cat((torch.flip(self.w, [0]), self.w), 0)
            sym_b = torch.cat((torch.flip(-self.b, [0]), self.b), 0)
            return sym_w, sym_b
        return self.w, self.b

    def update_state(self, device=None):
        w, b = self._get_params()
        n = (torch.sum(w) / 2).item()
        self.cum_w = torch.sub(torch.cumsum(torch.cat([torch.tensor([0.0]).to(w.device), w]), dim=0), n)
        self.average_points = (self.cum_w[1:] + self.cum_w[:-1]) / 2
        self.distance_points = (self.cum_w[1:] - self.cum_w[:-1]) / 2
        if device:
            self.to(device)

    def forward(self, x, beta=1.0, spatial_offset=None):
        """
        优化后的显存友好型 forward
        """
        w, b = self._get_params()
        b = torch.sort(b)[0]
        
        # 初始化输出张量
        y_out = torch.zeros_like(x)
        
        # 🟢 核心修复：通过循环 $L$ 次来替代巨大的 5D 广播
        # 这能极大降低显存峰值，且由于 L 较小 (64-128)，对训练速度影响忽略不计
        for i in range(len(w)):
            # 拿到当前阶梯的参数
            w_i = w[i]
            b_i = b[i]
            
            # 应用空间偏移
            if spatial_offset is not None:
                curr_b = b_i + spatial_offset # [B, 1, H, W]
            else:
                curr_b = b_i
            
            if beta == -1:
                # 硬量化累加
                y_out = y_out + (w_i / 2) * torch.sign(x - curr_b)
            else:
                # 软量化累加
                y_out = y_out + (w_i / 2) * torch.tanh(beta * (x - curr_b))

        # 计算退火误差 Et
        et = torch.tensor(0.0).to(x.device)
        if self.training and beta != -1:
            with torch.no_grad():
                # 硬量化用于误差计算
                y_hard = torch.zeros_like(x)
                for i in range(len(w)):
                    curr_b = b[i] + (spatial_offset if spatial_offset is not None else 0)
                    y_hard += (w[i] / 2) * torch.sign(x - curr_b)
            
            # 计算 MSE 差值
            e_soft = torch.mean((y_out - x)**2)
            e_hard = torch.mean((y_hard - x)**2)
            et = torch.abs(e_hard - e_soft)

        return y_out, et

# --- 2. 核心熵模型 (完全对齐原文 GaussianConditionalSoS) ---
class GaussianConditionalSoS(nn.Module):
    def __init__(self, channels, num_sigmoids, extrema):
        super().__init__()
        self.lower_bound_scale = LowerBound(0.11)
        self.sos = STanHQuantizer(channels, num_sigmoids, extrema)

    def _standardized_cumulative(self, inputs):
        return 0.5 * torch.erfc(-(2**-0.5) * torch.clamp(inputs, -20, 20))

    def define_v0_and_v1(self, values, avg_pts, dist_pts):
        """ 核心：原文算法计算点到动态边界的距离 """
        # 展平以便处理
        val_shape = values.shape
        v_flat = values.reshape(-1, 1)
        
        # 构造边界查找
        avg_left = torch.cat([torch.tensor([-100.0]).to(values.device), avg_pts])
        avg_right = torch.cat([avg_pts, torch.tensor([100.0]).to(values.device)])
        dist_left = torch.cat([torch.tensor([0.0]).to(values.device), dist_pts])
        dist_right = torch.cat([dist_pts, torch.tensor([0.0]).to(values.device)])

        # 寻找落入的区间
        mask = torch.logical_and(v_flat > avg_left, v_flat <= avg_right)
        v0 = torch.sum(dist_left * mask, dim=1).reshape(val_shape)
        v1 = torch.sum(dist_right * mask, dim=1).reshape(val_shape)
        return v0, v1

    def _likelihood(self, inputs, scales, means=None):
        self.sos.update_state(inputs.device)
        values = inputs - means if means is not None else inputs
        v0, v1 = self.define_v0_and_v1(values, self.sos.average_points, self.sos.distance_points)
        
        scales = self.lower_bound_scale(scales)
        # 对应原文公式：Phi((v1-val)/s) - Phi((-v0-val)/s)
        # 注意：这里的 values 已经减去 means，所以公式简化
        upper = self._standardized_cumulative((v1 - values) / scales)
        lower = self._standardized_cumulative((-v0 - values) / scales)
        return torch.clamp(upper - lower, min=1e-9)

    def forward(self, inputs, scales, means=None, beta=1.0, spatial_offset=None):
        # 1. 量化
        residual = inputs - means if means is not None else inputs
        y_q = self.sos(residual, beta, spatial_offset)
        y_hat = y_q + (means if means is not None else 0)
        # 2. 似然
        likelihood = self._likelihood(y_hat, scales, means)
        return y_hat, likelihood
    

class Compression(CompressionModel):
    def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, codebook_size, num_stanh=1):
        super().__init__()
        self.num_stanh = num_stanh
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.gain = 1
        self.M = M 

        self.encoder = Encoder(in_nc, M)
        self.encoder_y = Encoder(in_nc, M)
        self.hyper_enc = HyperEncoder(N, M)
        self.hyper_dec = HyperDecoder(N, M)
        self.decoder = Decoder(M)
        self.decoder_y = Decoder(M)
        self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
        self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)

        self.stanh_bank = nn.ModuleList([
            STanHQuantizer(
                channels=self.M,     
                num_sigmoids=10, 
                extrema=5.0, 
                symmetry=True
            ) for _ in range(num_stanh)
        ])

        self.predictor = EncoderPredictor(M)
        self.proj_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.local_context = nn.ModuleList(nn.Conv2d(slice_ch[i], slice_ch[i] * 2, 5, 1, 2) for i in range(len(slice_ch)))
        self.channel_context = nn.ModuleList(ChannelContextEX(sum(slice_ch[:i]), slice_ch[i] * 2) if i else None for i in range(slice_num))
        self.entropy_parameters_anchor = nn.ModuleList(EntropyParametersEX(M * 2 + (slice_ch[i] * 2 if i else 0), slice_ch[i] * 2) for i in range(slice_num))
        self.entropy_parameters_nonanchor = nn.ModuleList(EntropyParametersEX(M * 2 + slice_ch[i] * (4 if i else 2), slice_ch[i] * 2) for i in range(slice_num))

        self.codebook_size = codebook_size
        self.quantize = VectorQuantiser(self.codebook_size, N, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)


    def get_embedding(self, feat):
        emb = self.proj_head(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def side_encode(self, h_i):
        f_y = self.encoder_y(h_i)
        f_y = self.decoder_y(f_y)
        z_y = self.out_y(f_y)
        return f_y, z_y

    def get_interpolated_params(self, level):
        level = max(0.0, min(float(level), self.num_stanh - 1))
        floor_idx = math.floor(level)
        ceil_idx = math.ceil(level)
        decimal = level - floor_idx

        w1, b1 = self.stanh_bank[floor_idx]._get_params()
        if floor_idx == ceil_idx:
            return w1, b1
        
        w2, b2 = self.stanh_bank[ceil_idx]._get_params()
        w_interp = w1 * (1 - decimal) + w2 * decimal
        b_interp = b1 * (1 - decimal) + b2 * decimal
        return w_interp, b_interp

    def _standardized_cumulative(self, inputs):
        return 0.5 * torch.erfc(-(2**-0.5) * torch.clamp(inputs, -20, 20))

    def forward(self, x, h_y, beta=1.0, stanh_level=0.0):
        y = self.encoder(x)
        h_y = self.encoder_y(h_y)

        pred_mask = self.predictor(y)   # ✔ 保留
        y_scaled = y * self.gain

        z = self.hyper_enc(y)
        z_q, emb_loss, _ = self.quantize(z)
        hyper_params = self.hyper_dec(z_q)

        # =========================
        # 🔥 插值 STanH 参数
        # =========================
        w_v, b_v = self.get_interpolated_params(stanh_level)
        b_v = torch.sort(b_v)[0]

        device = x.device
        w_v = w_v.to(device)
        b_v = b_v.to(device)

        # =========================
        # 🔥 计算 STanH boundaries（关键）
        # =========================
        with torch.no_grad():
            n = (torch.sum(w_v) / 2)
            cum_w = torch.cumsum(torch.cat([torch.tensor([0.0], device=device), w_v]), dim=0) - n
            levels = cum_w
            boundaries = (levels[1:] + levels[:-1]) / 2

            dist_left = levels[1:] - boundaries
            dist_right = boundaries - levels[:-1]

            # dist_left = torch.cat([torch.tensor([10.0], device=device), dist_left])
            # dist_right = torch.cat([dist_right, torch.tensor([10.0], device=device)])

        # reshape for broadcast
        dl = dist_left.view(-1, 1, 1, 1, 1)
        dr = dist_right.view(-1, 1, 1, 1, 1)

        # expand STanH params
        w_v = w_v.view(-1, 1, 1, 1, 1)
        b_v = b_v.view(-1, 1, 1, 1, 1)

        y_hat_slices, y_likelihoods, q_likelihoods = [], [], []
        total_et = torch.zeros(1, device=x.device)
        slice_start = 0

        for idx in range(self.slice_num):
            ch = self.slice_ch[idx]
            y_slice = y_scaled[:, slice_start: slice_start + ch, ...]
            slice_start += ch

            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            # =========================
            # Anchor
            # =========================
            if idx == 0:
                params_a = self.entropy_parameters_anchor[idx](hyper_params)
            else:
                params_a = self.entropy_parameters_anchor[idx](torch.cat([
                    self.channel_context[idx](torch.cat(y_hat_slices, 1)),
                    hyper_params
                ], 1))

            mu_a, scale_a = params_a.chunk(2, 1)
            mu_a, scale_a = ckbd_anchor(mu_a), ckbd_anchor(scale_a)

            # ✅ 正确用 pred_mask（调 residual，不动 b）
            res_a = (slice_anchor - mu_a) * (1 + pred_mask)

            res_a = res_a.unsqueeze(0)
            y_q_a = torch.sum((w_v / 2) * torch.tanh(beta * (res_a - b_v)), dim=0)
            y_hat_a = y_q_a + mu_a

            # =========================
            # Non-anchor
            # =========================
            local_ctx = self.local_context[idx](y_hat_a)

            if idx == 0:
                params_na = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], 1)
                )
            else:
                params_na = self.entropy_parameters_nonanchor[idx](torch.cat([
                    local_ctx,
                    self.channel_context[idx](torch.cat(y_hat_slices, 1)),
                    hyper_params
                ], 1))

            mu_na, scale_na = params_na.chunk(2, 1)
            mu_na, scale_na = ckbd_nonanchor(mu_na), ckbd_nonanchor(scale_na)

            mu_slice = ckbd_merge(mu_a, mu_na)
            scale_slice = ckbd_merge(scale_a, scale_na)

            # =========================
            # 🔥 STanH quantization
            # =========================
            residual = (y_slice - mu_slice) * (1 + pred_mask)

            residual_expanded = residual.unsqueeze(0).expand(len(w_v), -1, -1, -1, -1)

            y_q = torch.sum((w_v / 2) * torch.tanh(beta * (residual_expanded - b_v)), dim=0)
            y_hat_slice = y_q + mu_slice

            # =========================
            # 🔥 正确 likelihood（核心修复）
            # =========================
            scales = self.gaussian_conditional.lower_bound_scale(scale_slice)

            L = w_v.shape[0]

            # residual_expanded = residual.unsqueeze(0).expand(L, -1, -1, -1, -1)
            scales_expanded = scales.unsqueeze(0).expand(L, -1, -1, -1, -1)

            upper = self._standardized_cumulative((dr - residual_expanded) / scales_expanded)
            lower = self._standardized_cumulative((-dl - residual_expanded) / scales_expanded)

            probs = torch.clamp(upper - lower, min=1e-9)

            y_lik = torch.sum(probs, dim=0)
            probs = torch.clamp(upper - lower, min=1e-9)
            y_lik = torch.sum(probs, dim=0)

            # 原始 likelihood（监控用）
            _, q_lik = self.gaussian_conditional(y_slice, scale_slice, mu_slice, False)

            y_hat_slices.append(y_hat_slice)
            y_likelihoods.append(y_lik)
            q_likelihoods.append(q_lik)
            if self.training and beta != -1:
                with torch.no_grad():
                    y_hard = torch.sum((w_v / 2) * torch.sign(residual_expanded - b_v), dim=0)

                e_soft = torch.mean((y_q - residual)**2)
                e_hard = torch.mean((y_hard - residual)**2)
                et = torch.abs(e_hard - e_soft)

                total_et = total_et + et

        y_hat = torch.cat(y_hat_slices, 1) / self.gain

        guide_hint = self.decoder(y_hat)
        output = self.out(guide_hint)
        h_y_out = self.decoder_y(h_y)

        avg_et = total_et / self.slice_num if self.slice_num > 0 else torch.tensor(0.0).to(x.device)

        return output, [torch.cat(y_likelihoods, 1)], [torch.cat(q_likelihoods, 1)], emb_loss, guide_hint, h_y_out, avg_et, pred_mask