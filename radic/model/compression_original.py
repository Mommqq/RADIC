

#original---------------------------------------------------------------------
import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils1.func import get_scale_table
from model.compression_modules import *

class Compression(CompressionModel):
    def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, codebook_size):
        super().__init__()

        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.encoder = Encoder(in_nc, M)
        self.encoder_y = Encoder(in_nc, M)
        self.hyper_enc = HyperEncoder(N, M)
        self.hyper_dec = HyperDecoder(N, M)
        self.decoder = Decoder(M)
        self.decoder_y = Decoder(M)
        self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
        self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)  
        )
        self.local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )

        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2) if i else None
            for i in range(slice_num)
        )

        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2)
            for i in range(slice_num)
        )

        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
            for i in range(slice_num)
        )

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

    def forward(self, x, h_y):
        # print("x:",x.shape) # (b,512,32,32)
        y = self.encoder(x)
        h_y = self.encoder_y(h_y)
        z = self.hyper_enc(y)
        z_q, emb_loss, _  = self.quantize(z)

        # Hyper-parameters
        hyper_params = self.hyper_dec(z_q)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            """
            Split y to anchor and non-anchor
            anchor :
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
            non-anchor:
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
            """
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
            
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        q_likelihoods = torch.cat(q_likelihoods, dim=1)
        
        guide_hint = self.decoder(y_hat)
        h_y = self.decoder_y(h_y)
        output = self.out(guide_hint)
        # output_y = self.out(h_y)
        return output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y

    # ---------------------------------------------------------------------
    # Compress & Decompress 
    # ---------------------------------------------------------------------
    def compress(self, x):
        y = self.encoder(x)
        z = self.hyper_enc(y)
        z_q, encoding_indices = self.quantize.quant(z)
        
        torch.backends.cudnn.deterministic = True
        z_strings = compress_hyper_latent(encoding_indices, self.codebook_size)
        hyper_params = self.hyper_dec(z_q)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False
        return {
            "strings": [y_strings, [z_strings]],
            "shape": z.size()[-2:]
        }
    
    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True

        y_strings = strings[0][0]
        z_strings = strings[1][0]
        encoding_indices = decompress_hyper_latent(z_strings, shape, codebook_size=self.codebook_size)
        z_q = self.quantize.get_codebook_entry(encoding_indices.long())
        
        hyper_params = self.hyper_dec(z_q)

        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)
            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False

        guide_hint = self.decoder(y_hat)

        output = self.out(guide_hint)

        return output, guide_hint
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
#----------------------------------------------original--------------------------------------------------------------------

#--------------------------------------Stanh--------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from compressai.models import CompressionModel
# from compressai.entropy_models import GaussianConditional
# from model.compression_modules import * 

# class STanHQuantizer(nn.Module):
#     def __init__(self, num_sigmoids=64, extrema=32, symmetry=True):
#         super().__init__()
#         self.num_sigmoids = int(num_sigmoids)
#         self.symmetry = symmetry
#         self.extrema = extrema
#         self.jump = float(extrema) / num_sigmoids if num_sigmoids > 0 else 1.0
        
#         if self.symmetry:
#             self.w = nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump)
#             initial_b = torch.arange(self.jump/2, extrema + self.jump/2, self.jump)
#             self.b = nn.Parameter(initial_b[:self.num_sigmoids])
#         else:
#             self.w = nn.Parameter(torch.zeros(self.num_sigmoids * 2) + self.jump)
#             self.b = nn.Parameter(torch.arange(-extrema + 0.5, extrema + 0.5, self.jump))

#     def _get_params(self):
#         if self.symmetry:
#             sym_w = torch.cat((torch.flip(self.w, [0]), self.w), 0)
#             sym_b = torch.cat((torch.flip(-self.b, [0]), self.b), 0)
#             return sym_w, sym_b
#         return self.w, self.b


#     def forward(self, x, beta=1.0):
#         """
#         x: [B, C, H, W]
#         w, b: [L] (L=128)
#         """
#         w, b = self._get_params()
#         b = torch.sort(b)[0]
        
#         # --- 核心修复：调整维度以适配广播机制 ---
#         # 我们需要将 w 和 b 扩展为 [L, 1, 1, 1, 1]
#         # 将 x 扩展为 [1, B, C, H, W]
#         # 这样运算结果就是 [L, B, C, H, W]，然后对第 0 维 (L) 求和
        
#         w = w.view(-1, 1, 1, 1, 1).to(x.device)
#         b = b.view(-1, 1, 1, 1, 1).to(x.device)
#         x_expanded = x.unsqueeze(0) # 变为 [1, B, C, H, W]

#         if beta == -1:
#             # 硬量化 (推理)
#             # [L, B, C, H, W] -> sum over dim 0 -> [B, C, H, W]
#             y_quantized = torch.sum((w / 2) * torch.sign(x_expanded - b), dim=0)
#             return y_quantized, torch.tensor(0.0).to(x.device)
#         else:
#             # 软量化 (训练)
#             # [L, B, C, H, W] -> sum over dim 0 -> [B, C, H, W]
#             y_soft = torch.sum((w / 2) * torch.tanh(beta * (x_expanded - b)), dim=0)

#             # 计算 Et 用于退火 (Equation 7)
#             et = torch.tensor(0.0).to(x.device)
#             if self.training:
#                 # 计算硬量化版本以对比误差
#                 with torch.no_grad():
#                     y_hard = torch.sum((w / 2) * torch.sign(x_expanded - b), dim=0)
                
#                 # Eq. 5 & 6
#                 e_soft = torch.mean((y_soft - x)**2)
#                 e_hard = torch.mean((y_hard - x)**2)
#                 et = torch.abs(e_hard - e_soft)

#             return y_soft, et

# class Compression(CompressionModel):
#     def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, codebook_size):
#         super().__init__()

#         self.slice_num = slice_num
#         self.slice_ch = slice_ch

#         # --- 核心网络 ---
#         self.encoder = Encoder(in_nc, M)
#         self.encoder_y = Encoder(in_nc, M)
#         self.hyper_enc = HyperEncoder(N, M)
#         self.hyper_dec = HyperDecoder(N, M)
#         self.decoder = Decoder(M)
#         self.decoder_y = Decoder(M)
#         self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
#         self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)

#         # --- STanH 量化器 ---
#         self.stanh_y = STanHQuantizer(num_sigmoids=64, extrema=32, symmetry=True)
#         self.proj_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 256),
#             nn.SiLU(),
#             nn.Linear(256, 256)  
#         )
#         # --- 上下文模型与参数估计 ---
#         self.local_context = nn.ModuleList(
#             nn.Conv2d(slice_ch[i], slice_ch[i] * 2, 5, 1, 2) for i in range(len(slice_ch))
#         )
#         self.channel_context = nn.ModuleList(
#             ChannelContextEX(sum(slice_ch[:i]), slice_ch[i] * 2) if i else None
#             for i in range(slice_num)
#         )
#         self.entropy_parameters_anchor = nn.ModuleList(
#             EntropyParametersEX(M * 2 + (slice_ch[i] * 2 if i else 0), slice_ch[i] * 2)
#             for i in range(slice_num)
#         )
#         self.entropy_parameters_nonanchor = nn.ModuleList(
#             EntropyParametersEX(M * 2 + slice_ch[i] * (4 if i else 2), slice_ch[i] * 2)
#             for i in range(slice_num)
#         )

#         self.codebook_size = codebook_size
#         self.quantize = VectorQuantiser(self.codebook_size, N, contras_loss=True)
#         self.gaussian_conditional = GaussianConditional(None)

#     def get_embedding(self, feat):
#         emb = self.proj_head(feat)
#         return torch.nn.functional.normalize(emb, p=2, dim=1)

#     def side_encode(self, h_i):
#         f_y = self.encoder_y(h_i)
#         f_y = self.decoder_y(f_y)
#         z_y = self.out_y(f_y)
#         return f_y, z_y


#     def forward(self, x, h_y, beta=1.0):
#         y = self.encoder(x)
#         h_y = self.encoder_y(h_y)
        
#         # 1. 超验路径 (VQ 保持不变)
#         z = self.hyper_enc(y)
#         z_q, emb_loss, _  = self.quantize(z)
#         hyper_params = self.hyper_dec(z_q)

#         # 2. 基于 STanH 的主路径切片量化
#         y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
#         y_hat_slices = []
#         y_likelihoods = []
#         q_likelihoods = []
        
#         # 用于累积本步所有量化操作的 Et
#         total_et = 0.0
#         quant_steps = 0

#         for idx, y_slice in enumerate(y_slices):
#             slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            
#             # --- A. 处理 Anchor ---
#             if idx == 0:
#                 params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
#             else:
#                 channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
#                 params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
            
#             scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#             scales_anchor = ckbd_anchor(scales_anchor)
#             means_anchor = ckbd_anchor(means_anchor)
            
#             # STanH 量化 Anchor
#             residual_anchor = slice_anchor - means_anchor
#             quant_anchor, et_a = self.stanh_y(residual_anchor, beta=beta)
#             slice_anchor = quant_anchor + means_anchor
            
#             # 累计误差
#             total_et += et_a
#             quant_steps += 1
            
#             # --- B. 处理 Non-anchor ---
#             if idx == 0:
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
#             else:
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
            
#             scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#             scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
#             means_nonanchor = ckbd_nonanchor(means_nonanchor)
            
#             scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
#             means_slice = ckbd_merge(means_anchor, means_nonanchor)
            
#             # 熵模型计算
#             _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
#             _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
            
#             # STanH 量化 Non-anchor
#             residual_nonanchor = slice_nonanchor - means_nonanchor
#             quant_nonanchor, et_na = self.stanh_y(residual_nonanchor, beta=beta)
#             slice_nonanchor = quant_nonanchor + means_nonanchor
            
#             # 累计误差
#             total_et += et_na
#             quant_steps += 1
            
#             y_hat_slice = slice_anchor + slice_nonanchor
#             y_hat_slices.append(y_hat_slice)
#             y_likelihoods.append(y_slice_likelihoods)
#             q_likelihoods.append(q_slice_likelihoods)

#         # 3. 结果整合
#         y_hat = torch.cat(y_hat_slices, dim=1)
#         y_likelihoods = torch.cat(y_likelihoods, dim=1)
#         q_likelihoods = torch.cat(q_likelihoods, dim=1)
        
#         guide_hint = self.decoder(y_hat)
#         h_y_out = self.decoder_y(h_y)
#         output = self.out(guide_hint)
        
#         # 计算平均 Et (Equation 7 & 8 的基础)
#         avg_et = total_et / quant_steps if quant_steps > 0 else torch.tensor(0.0).to(x.device)
        
#         return output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y_out, avg_et

#     def update(self, scale_table=None, force=False):
#         if scale_table is None:
#             from utils1.func import get_scale_table
#             scale_table = get_scale_table()
#         updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
#         updated |= super().update(force=force)
#         return updated