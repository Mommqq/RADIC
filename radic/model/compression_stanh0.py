

#original---------------------------------------------------------------------
import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils1.func import get_scale_table
from model.compression_modules import *

# class Compression(CompressionModel):
#     def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, codebook_size):
#         super().__init__()

#         self.slice_num = slice_num
#         self.slice_ch = slice_ch

#         self.encoder = Encoder(in_nc, M)
#         self.encoder_y = Encoder(in_nc, M)
#         self.hyper_enc = HyperEncoder(N, M)
#         self.hyper_dec = HyperDecoder(N, M)
#         self.decoder = Decoder(M)
#         self.decoder_y = Decoder(M)
#         self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
#         self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)
#         self.proj_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, 256),
#             nn.SiLU(),
#             nn.Linear(256, 256)  
#         )
#         self.local_context = nn.ModuleList(
#             nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
#             for i in range(len(slice_ch))
#         )

#         self.channel_context = nn.ModuleList(
#             ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2) if i else None
#             for i in range(slice_num)
#         )

#         # Use channel_ctx and hyper_params
#         self.entropy_parameters_anchor = nn.ModuleList(
#             EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
#             if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2)
#             for i in range(slice_num)
#         )

#         # Entropy parameters for non-anchors
#         # Use spatial_params, channel_ctx and hyper_params
#         self.entropy_parameters_nonanchor = nn.ModuleList(
#             EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2)
#             if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
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

#     def forward(self, x, h_y):
#         # print("x:",x.shape) # (b,512,32,32)
#         y = self.encoder(x)
#         h_y = self.encoder_y(h_y)
#         z = self.hyper_enc(y)
#         z_q, emb_loss, _  = self.quantize(z)

#         # Hyper-parameters
#         hyper_params = self.hyper_dec(z_q)

#         y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
#         y_hat_slices = []
#         y_likelihoods = []
#         q_likelihoods = []
#         for idx, y_slice in enumerate(y_slices):
#             """
#             Split y to anchor and non-anchor
#             anchor :
#                 0 1 0 1 0
#                 1 0 1 0 1
#                 0 1 0 1 0
#                 1 0 1 0 1
#                 0 1 0 1 0
#             non-anchor:
#                 1 0 1 0 1
#                 0 1 0 1 0
#                 1 0 1 0 1
#                 0 1 0 1 0
#                 1 0 1 0 1
#             """
#             slice_anchor, slice_nonanchor = ckbd_split(y_slice)
#             if idx == 0:
#                 # Anchor
#                 params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
#                 scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#                 # split means and scales of anchor
#                 scales_anchor = ckbd_anchor(scales_anchor)
#                 means_anchor = ckbd_anchor(means_anchor)
#                 # round anchor
#                 slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
#                 # Non-anchor
#                 # local_ctx: [B, H, W, 2 * C]
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
#                 scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#                 # split means and scales of nonanchor
#                 scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
#                 means_nonanchor = ckbd_nonanchor(means_nonanchor)
#                 # merge means and scales of anchor and nonanchor
#                 scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
#                 means_slice = ckbd_merge(means_anchor, means_nonanchor)
#                 _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
#                 _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
#                 # round slice_nonanchor
#                 slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
#                 y_hat_slice = slice_anchor + slice_nonanchor
#                 y_hat_slices.append(y_hat_slice)
#                 y_likelihoods.append(y_slice_likelihoods)
#                 q_likelihoods.append(q_slice_likelihoods)
#             else:
#                 channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
#                 # Anchor(Use channel context and hyper params)
#                 params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
#                 scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#                 # split means and scales of anchor
#                 scales_anchor = ckbd_anchor(scales_anchor)
#                 means_anchor = ckbd_anchor(means_anchor)
#                 # round anchor
#                 slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
#                 # Non-anchor
#                 # ctx_params: [B, H, W, 2 * C]
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
#                 scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#                 # split means and scales of nonanchor
#                 scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
#                 means_nonanchor = ckbd_nonanchor(means_nonanchor)
#                 # merge means and scales of anchor and nonanchor
#                 scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
#                 means_slice = ckbd_merge(means_anchor, means_nonanchor)
#                 _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
#                 _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
#                 # round slice_nonanchor
#                 slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
#                 y_hat_slice = slice_anchor + slice_nonanchor
#                 y_hat_slices.append(y_hat_slice)
#                 y_likelihoods.append(y_slice_likelihoods)
#                 q_likelihoods.append(q_slice_likelihoods)

#         y_hat = torch.cat(y_hat_slices, dim=1)
#         y_likelihoods = torch.cat(y_likelihoods, dim=1)
#         q_likelihoods = torch.cat(q_likelihoods, dim=1)
        
#         guide_hint = self.decoder(y_hat)
#         h_y = self.decoder_y(h_y)
#         output = self.out(guide_hint)
#         # output_y = self.out(h_y)
#         return output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y

#     # ---------------------------------------------------------------------
#     # Compress & Decompress 
#     # ---------------------------------------------------------------------
#     def compress(self, x):
#         y = self.encoder(x)
#         z = self.hyper_enc(y)
#         z_q, encoding_indices = self.quantize.quant(z)
        
#         torch.backends.cudnn.deterministic = True
#         z_strings = compress_hyper_latent(encoding_indices, self.codebook_size)
#         hyper_params = self.hyper_dec(z_q)

#         y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
#         y_hat_slices = []

#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
#         offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
#         encoder = BufferedRansEncoder()
#         symbols_list = []
#         indexes_list = []
#         y_strings = []

#         for idx, y_slice in enumerate(y_slices):
#             slice_anchor, slice_nonanchor = ckbd_split(y_slice)
#             if idx == 0:
#                 # Anchor
#                 params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
#                 scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#                 # round and compress anchor
#                 slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
#                 # Non-anchor
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
#                 scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#                 # round and compress nonanchor
#                 slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
#                 y_slice_hat = slice_anchor + slice_nonanchor
#                 y_hat_slices.append(y_slice_hat)

#             else:
#                 # Anchor
#                 channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
#                 params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
#                 scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#                 # round and compress anchor
#                 slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
#                 # Non-anchor
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
#                 scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#                 # round and compress nonanchor
#                 slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
#                 y_hat_slices.append(slice_nonanchor + slice_anchor)

#         encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
#         y_string = encoder.flush()
#         y_strings.append(y_string)

#         torch.backends.cudnn.deterministic = False
#         return {
#             "strings": [y_strings, [z_strings]],
#             "shape": z.size()[-2:]
#         }
    
#     def decompress(self, strings, shape):
#         torch.backends.cudnn.deterministic = True

#         y_strings = strings[0][0]
#         z_strings = strings[1][0]
#         encoding_indices = decompress_hyper_latent(z_strings, shape, codebook_size=self.codebook_size)
#         z_q = self.quantize.get_codebook_entry(encoding_indices.long())
        
#         hyper_params = self.hyper_dec(z_q)

#         y_hat_slices = []

#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
#         offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
#         decoder = RansDecoder()
#         decoder.set_stream(y_strings)

#         for idx in range(self.slice_num):
#             if idx == 0:
#                 # Anchor
#                 params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
#                 scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#                 # decompress anchor
#                 slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
#                 # Non-anchor
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
#                 scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#                 # decompress non-anchor
#                 slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
#                 y_hat_slice = slice_nonanchor + slice_anchor
#                 y_hat_slices.append(y_hat_slice)
#             else:
#                 # Anchor
#                 channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
#                 params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
#                 scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#                 # decompress anchor
#                 slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
#                 # Non-anchor
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
#                 scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#                 # decompress non-anchor
#                 slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
#                 y_hat_slice = slice_nonanchor + slice_anchor
#                 y_hat_slices.append(y_hat_slice)

#         y_hat = torch.cat(y_hat_slices, dim=1)
#         torch.backends.cudnn.deterministic = False

#         guide_hint = self.decoder(y_hat)

#         output = self.out(guide_hint)

#         return output, guide_hint
    
#     def update(self, scale_table=None, force=False):
#         if scale_table is None:
#             scale_table = get_scale_table()
#         updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
#         updated |= super().update(force=force)
#         return updated
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

#     def forward(self, x, beta=1.0, spatial_offset=None):
#         """
#         x: [B, C, H, W]
#         spatial_offset: [B, 1, H, W] 
#         """
#         # print("z:",x.max(), x.min())
#         w, b = self._get_params()
#         b = torch.sort(b)[0]
        
#         # 基础参数维度: [L, 1, 1, 1, 1]
#         w = w.view(-1, 1, 1, 1, 1).to(x.device)
#         b_base = b.view(-1, 1, 1, 1, 1).to(x.device)
#         x_expanded = x.unsqueeze(0) # [1, B, C, H, W]

#         # 如果传入了空间偏移，将其应用到量化间隔 b 上
#         # 原理：预测越准的地方，量化间隔可以发生偏移或扩大，从而影响码率分配
#         if spatial_offset is not None:
#             # b 变为 [L, B, 1, H, W]
#             # 这里通过加法改变 quantization bins 的位置
#             b = b_base + spatial_offset.unsqueeze(0)
#         else:
#             b = b_base

#         if beta == -1:
#             y_quantized = torch.sum((w / 2) * torch.sign(x_expanded - b), dim=0)
#             return y_quantized, torch.tensor(0.0).to(x.device)
#         else:
#             y_soft = torch.sum((w / 2) * torch.tanh(beta * (x_expanded - b)), dim=0)    
#             et = torch.tensor(0.0).to(x.device)
#             if self.training:
#                 with torch.no_grad():
#                     y_hard = torch.sum((w / 2) * torch.sign(x_expanded - b), dim=0)
#                 e_soft = torch.mean((y_soft - x)**2)
#                 e_hard = torch.mean((y_hard - x)**2)
#                 et = torch.abs(e_hard - e_soft)
#             return y_soft, et

# class EncoderPredictor(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(channels // 4, 1, 3, 1, 1),
#             nn.Sigmoid() #
#         )

#     def forward(self, y):
#         return self.net(y)

# class Compression(CompressionModel):
#     def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, codebook_size):
#         super().__init__()
#         self.slice_num = slice_num
#         self.slice_ch = slice_ch

#         self.encoder = Encoder(in_nc, M)
#         self.encoder_y = Encoder(in_nc, M)
#         self.hyper_enc = HyperEncoder(N, M)
#         self.hyper_dec = HyperDecoder(N, M)
#         self.decoder = Decoder(M)
#         self.decoder_y = Decoder(M)
#         self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
#         self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)

#         self.stanh_y = STanHQuantizer(num_sigmoids=16, extrema=4, symmetry=True)
#         self.predictor = EncoderPredictor(M)

#         self.proj_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
#             nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256)  
#         )
        
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
        
#         pred_mask = self.predictor(y) 
        
#         z = self.hyper_enc(y)
#         z_q, emb_loss, _  = self.quantize(z)
#         hyper_params = self.hyper_dec(z_q)

#         y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
#         y_hat_slices = []
#         y_likelihoods = []
#         q_likelihoods = []
        
#         total_et = 0.0
#         quant_steps = 0

#         for idx, y_slice in enumerate(y_slices):
#             slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            
#             # --- Anchor ---
#             if idx == 0:
#                 params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
#             else:
#                 channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
#                 params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
            
#             scales_anchor, means_anchor = params_anchor.chunk(2, 1)
#             scales_anchor, means_anchor = ckbd_anchor(scales_anchor), ckbd_anchor(means_anchor)
            
#             residual_anchor = slice_anchor - means_anchor
#             quant_anchor, et_a = self.stanh_y(residual_anchor, beta=beta, spatial_offset=pred_mask)
#             slice_anchor = quant_anchor + means_anchor
#             total_et += et_a
#             quant_steps += 1
            
#             # --- Non-anchor ---
#             if idx == 0:
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
#             else:
#                 local_ctx = self.local_context[idx](slice_anchor)
#                 params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
            
#             scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
#             scales_nonanchor, means_nonanchor = ckbd_nonanchor(scales_nonanchor), ckbd_nonanchor(means_nonanchor)
            
#             scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
#             means_slice = ckbd_merge(means_anchor, means_nonanchor)
                     
#             residual_nonanchor = slice_nonanchor - means_nonanchor
#             quant_nonanchor, et_na = self.stanh_y(residual_nonanchor, beta=beta, spatial_offset=pred_mask)
#             slice_nonanchor = quant_nonanchor + means_nonanchor
#             total_et += et_na
#             quant_steps += 1
 
 
#             y_hat_slice = slice_anchor + slice_nonanchor
#             # print("y_hat_slice:", y_hat_slice.max(),y_hat_slice.min())
#             _, y_slice_likelihoods = self.gaussian_conditional(y_hat_slice, scales_slice, means_slice)
#             _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False) 
 
            
#             y_hat_slices.append(slice_anchor + slice_nonanchor)
#             y_likelihoods.append(y_slice_likelihoods)
#             q_likelihoods.append(q_slice_likelihoods)

#         y_hat = torch.cat(y_hat_slices, dim=1)
#         guide_hint = self.decoder(y_hat)
#         h_y_out = self.decoder_y(h_y)
#         output = self.out(guide_hint)
        
#         avg_et = total_et / quant_steps if quant_steps > 0 else torch.tensor(0.0).to(x.device)

#         return output, [torch.cat(y_likelihoods, dim=1)], [torch.cat(q_likelihoods, dim=1)], emb_loss, guide_hint, h_y_out, avg_et, pred_mask

#     def update(self, scale_table=None, force=False):
#         if scale_table is None:
#             from utils1.func import get_scale_table
#             scale_table = get_scale_table()
#         updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
#         updated |= super().update(force=force)
#         return updated



#-------------------------------------Stanh_pro----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from model.compression_modules import * 

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

# ==========================================
# 1. 增强型 STanH 量化器（提供边界信息）
# ==========================================
class STanHQuantizer(nn.Module):
    def __init__(self, num_sigmoids=64, extrema=32, symmetry=True):
        super().__init__()
        self.num_sigmoids = int(num_sigmoids)
        self.symmetry = symmetry
        self.extrema = extrema
        self.jump = float(extrema) / num_sigmoids if num_sigmoids > 0 else 1.0
        
        if self.symmetry:
            self.w = nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump)
            initial_b = torch.arange(self.jump/2, extrema + self.jump/2, self.jump)
            self.b = nn.Parameter(initial_b[:self.num_sigmoids])
        else:
            self.w = nn.Parameter(torch.zeros(self.num_sigmoids * 2) + self.jump)
            self.b = nn.Parameter(torch.arange(-extrema + 0.5, extrema + 0.5, self.jump))

    def _get_params(self):
        if self.symmetry:
            sym_w = torch.cat((torch.flip(self.w, [0]), self.w), 0)
            sym_b = torch.cat((torch.flip(-self.b, [0]), self.b), 0)
            return sym_w, sym_b
        return self.w, self.b

    def get_levels_and_boundaries(self):
        w, b = self._get_params()
        n = (torch.sum(w) / 2).item()
        cum_w = torch.zeros(len(w) + 1).to(w.device)
        cum_w[1:] = torch.cumsum(w, dim=0)
        levels = cum_w - n  
        boundaries = (levels[1:] + levels[:-1]) / 2
        dist_left = levels[1:] - boundaries
        dist_right = boundaries - levels[:-1]
        dist_left = torch.cat([torch.tensor([10.0]).to(w.device), dist_left])
        dist_right = torch.cat([dist_right, torch.tensor([10.0]).to(w.device)])
        return levels, boundaries, dist_left, dist_right

    def forward(self, x, beta=1.0, spatial_offset=None):
        w, b = self._get_params()
        b = torch.sort(b)[0]
        w = w.view(-1, 1, 1, 1, 1).to(x.device)
        b_base = b.view(-1, 1, 1, 1, 1).to(x.device)
        
        gain = 1
        x_scaled = x * gain
        x_expanded = x_scaled.unsqueeze(0)

        b_final = b_base + (spatial_offset.unsqueeze(0) if spatial_offset is not None else 0)
        b_final = b_base 
        if beta == -1:
            y_q = torch.sum((w / 2) * torch.sign(x_expanded - b_final), dim=0)
            return y_q / gain, torch.tensor(0.0).to(x.device)
        else:
            y_q_soft = torch.sum((w / 2) * torch.tanh(beta * (x_expanded - b_final)), dim=0)
            y_soft = y_q_soft / gain
            et = torch.tensor(0.0).to(x.device)
            if self.training:
                with torch.no_grad():
                    y_hard = torch.sum((w / 2) * torch.sign(x_expanded - b_final), dim=0) / gain
                et = torch.abs(torch.mean((y_hard - x)**2) - torch.mean((y_soft - x)**2))
            return y_soft, et

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
        self.stanh_y = STanHQuantizer(num_sigmoids=64, extrema=5, symmetry=True)
        self.predictor = EncoderPredictor(M)
        self.proj_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.local_context = nn.ModuleList(nn.Conv2d(slice_ch[i], slice_ch[i]*2, 5, 1, 2) for i in range(len(slice_ch)))
        self.channel_context = nn.ModuleList(ChannelContextEX(sum(slice_ch[:i]), slice_ch[i]*2) if i else None for i in range(slice_num))
        self.entropy_parameters_anchor = nn.ModuleList(EntropyParametersEX(M*2 + (slice_ch[i]*2 if i else 0), slice_ch[i]*2) for i in range(slice_num))
        self.entropy_parameters_nonanchor = nn.ModuleList(EntropyParametersEX(M*2 + slice_ch[i]*(4 if i else 2), slice_ch[i]*2) for i in range(slice_num))
        self.codebook_size = codebook_size
        self.quantize = VectorQuantiser(self.codebook_size, N, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)

    # def _standardized_cumulative(self, inputs):
    #     return 0.5 * torch.erfc(-(2**-0.5) * inputs)


    def get_embedding(self, feat):
        emb = self.proj_head(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def side_encode(self, h_i):
        f_y = self.encoder_y(h_i)
        f_y = self.decoder_y(f_y)
        z_y = self.out_y(f_y)
        return f_y, z_y

    def _standardized_cumulative(self, inputs):
        safe_inputs = torch.clamp(inputs, min=-20.0, max=20.0)
        return 0.5 * torch.erfc(-(2**-0.5) * safe_inputs)

    def forward(self, x, h_y, beta=1.0):
        y = self.encoder(x)
        h_y = self.encoder_y(h_y)
        pred_mask = self.predictor(y) 

        z = self.hyper_enc(y)
        z_q, emb_loss, _  = self.quantize(z)
        hyper_params = self.hyper_dec(z_q)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices, y_likelihoods, q_likelihoods = [], [], []
        total_et, quant_steps = 0.0, 0

        # 获取当前 STanH 的平均步长用于 Likelihood 估计
        # 因为 STanH 是参数化的，我们假设平均 bin 宽为 1.0 / num_sigmoids
        half_bin = self.stanh_y.extrema / self.stanh_y.num_sigmoids

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            
            # --- Anchor 熵参数 ---
            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
            
            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor, means_anchor = ckbd_anchor(scales_anchor), ckbd_anchor(means_anchor)

            # STanH 量化 Anchor (为后续 context 提供输入)
            residual_anchor = slice_anchor - means_anchor
            y_q_anchor, et_a = self.stanh_y(residual_anchor, beta=beta, spatial_offset=pred_mask)
            slice_anchor_hat = y_q_anchor + means_anchor
            total_et += et_a
            quant_steps += 1

            # --- Non-anchor 熵参数 ---
            local_ctx = self.local_context[idx](slice_anchor_hat)
            if idx == 0:
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
            else:
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
            
            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor, means_nonanchor = ckbd_nonanchor(scales_nonanchor), ckbd_nonanchor(means_nonanchor)

            # --- 整体量化与似然计算 ---
            scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
            means_slice = ckbd_merge(means_anchor, means_nonanchor)
            
            # 1. 整体 STanH 量化
            residual_slice = y_slice - means_slice
            y_q_slice, et_slice = self.stanh_y(residual_slice, beta=beta, spatial_offset=pred_mask)
            y_slice_hat = y_q_slice + means_slice
            
            total_et += et_slice
            quant_steps += 1

            # 🟢 关键点 3：对齐似然计量的尺度
            # 使用 lower_bound 保护 scales
            scales = self.gaussian_conditional.lower_bound_scale(scales_slice)
            
            # 计算基于 STanH 尺度的似然
            # 原理：将积分范围收缩到实际量化阶梯的宽度
            # 这样 Bpp Loss 就会产生一个强大的“推力”，防止 w 塌缩
            upper = self._standardized_cumulative((half_bin - y_q_slice) / scales)
            lower = self._standardized_cumulative((-half_bin - y_q_slice) / scales)
            
            y_slice_likelihoods = torch.clamp(upper - lower, min=1e-9)
            
            # 代理似然保持不变
            _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)

            y_hat_slices.append(y_slice_hat)
            y_likelihoods.append(y_slice_likelihoods)
            q_likelihoods.append(q_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        guide_hint = self.decoder(y_hat)
        h_y_out = self.decoder_y(h_y)
        output = self.out(guide_hint)
        
        avg_et = total_et / quant_steps if quant_steps > 0 else torch.tensor(0.0).to(x.device)

        return output, [torch.cat(y_likelihoods, dim=1)], [torch.cat(q_likelihoods, dim=1)], emb_loss, guide_hint, h_y_out, avg_et, pred_mask

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            from utils1.func import get_scale_table
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated