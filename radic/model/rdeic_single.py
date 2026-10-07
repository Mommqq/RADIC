import torch
import torch as th
import torch.nn as nn
from torchvision.transforms import transforms
from typing import Dict, Mapping, Any
import faiss
import glob
import os
import torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3
import random
import os
import math
import pyiqa
import numpy as np
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from utils1.utils import *

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)

from .spaced_sampler_relay import SpacedSampler
from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from .lpips import LPIPS



class ResBlock1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, channels)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual

class UpsampleBlock1(nn.Module):
    """ PixelShuffle 上采样模块 """
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        return x



#-----------------------------------attn+warp--------------------------------------------------------------------------------------

# class SinePositionalEncoding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.cached_grid = None # 缓存
        
#     def forward(self, x):
#         B, C, H, W = x.shape
#         device = x.device
        
#         # 如果尺寸变了，或者还没缓存，才重新计算
#         if self.cached_grid is None or self.cached_grid.shape[1:3] != (H, W):
#             y_pos = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(1, H, W)
#             x_pos = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(1, H, W)
#             self.cached_grid = torch.stack([x_pos, y_pos], dim=-1) # [1, H, W, 2]
            
#         # 扩展 Batch 维度 (零开销)
#         return self.cached_grid.expand(B, -1, -1, -1)

# class WarpFusionBlock(nn.Module):
#     """
#     显式 Warp + 拼接融合模块
#     用于高分辨率层，高效搬运纹理
#     """
#     def __init__(self, channels):
#         super().__init__()
#         # 融合层: 输入通道翻倍(Main+Side)，输出回原通道
#         self.fusion = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 3, 1, 1),
#             nn.GroupNorm(32, channels),
#             nn.SiLU(),
#             nn.Conv2d(channels, channels, 3, 1, 1) 
#         )

#     def get_grid(self, B, H, W, device):
#         y_pos = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(B, H, W)
#         x_pos = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(B, H, W)
#         grid = torch.stack([x_pos, y_pos], dim=-1)
#         return grid

#     def forward(self, x, side, flow_field):
#         """
#         flow_field: [B, 2, H, W] (需要是当前分辨率)
#         """
#         B, C, H, W = x.shape
        
#         # 1. 确保 Flow 维度正确并生成 Grid
#         # flow_field 应该是 [B, 2, H, W]，转为 [B, H, W, 2] 给 grid_sample
#         flow = flow_field.permute(0, 2, 3, 1)
        
#         base_grid = self.get_grid(B, H, W, x.device)
#         sample_grid = base_grid + flow
        
#         # 2. 显式 Warp
#         warped_side = F.grid_sample(side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
#         # 3. 拼接融合
#         combined = torch.cat([x, warped_side], dim=1)
#         out = self.fusion(combined)
        
#         return x + out

# class GeometryCorrectedAttention(nn.Module):
#     """
#     几何矫正注意力模块
#     用于低分辨率层，利用 Flow 修正位置编码，进行软对齐
#     """
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim_head = dim // num_heads
#         self.scale = self.dim_head ** -0.5
        
#         self.to_q = nn.Conv2d(dim, dim, 1)
#         self.to_k = nn.Conv2d(dim, dim, 1)
#         self.to_v = nn.Conv2d(dim, dim, 1)
#         self.to_out = nn.Conv2d(dim, dim, 1)
        
#         self.pos_map = nn.Sequential(
#             nn.Linear(2, dim // 2),
#             nn.SiLU(),
#             nn.Linear(dim // 2, dim),
#         )
#         self.pe_generator = SinePositionalEncoding(dim)

#     def forward(self, x, side, flow):
#         B, C, H, W = x.shape
        
#         # 1. 位置编码计算
#         base_grid = self.pe_generator(x) # [B, H, W, 2]
        
#         # Key的位置 = 基础位置 + Flow偏移
#         # flow: [B, 2, H, W] -> [B, H, W, 2]
#         flow_permute = flow.permute(0, 2, 3, 1)
#         grid_k = base_grid + flow_permute
        
#         # 合并计算 PE map
#         combined_grid = torch.cat([base_grid, grid_k], dim=0)
#         combined_pe = self.pos_map(combined_grid).permute(0, 3, 1, 2)
#         pe_q, pe_k = torch.chunk(combined_pe, 2, dim=0)
        
#         # 2. 投影 (注入位置信息)
#         q = self.to_q(x + pe_q)
#         k = self.to_k(side + pe_k)
#         v = self.to_v(side)
        
#         # 3. Flash Attention
#         q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads).contiguous()
#         k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads).contiguous()
#         v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads).contiguous()
        
#         out = F.scaled_dot_product_attention(q, k, v)
#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
#         return self.to_out(out) + x

# # =========================================================================
# # 2. 混合高保真解码器 (Hybrid Fidelity Decoder)
# # =========================================================================

# class GeometricFidelityDecoder(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, base_channels=128, num_res_blocks=2):
#         super().__init__()
        
#         # --- Heads ---
#         self.head_main = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
#         self.head_side = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
#         # ==================== Stage 1: Attention (Low Res) ====================
#         # Level 0 (32x32): 使用注意力
#         self.fuse0 = GeometryCorrectedAttention(base_channels)
#         self.res0 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # Level 1 (32 -> 64): 使用注意力
#         self.up1_main = UpsampleBlock1(base_channels, scale_factor=2)
#         self.up1_side = UpsampleBlock1(base_channels, scale_factor=2)
#         self.fuse1 = GeometryCorrectedAttention(base_channels)
#         self.res1 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # ==================== Stage 2: Warp + Concat (High Res) ====================
#         # Level 2 (64 -> 128): 使用 Warp 拼接
#         self.up2_main = UpsampleBlock1(base_channels, scale_factor=2)
#         self.up2_side = UpsampleBlock1(base_channels, scale_factor=2)
#         self.fuse2 = WarpFusionBlock(base_channels)
#         self.res2 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # Level 3 (128 -> 256): 使用 Warp 拼接
#         self.up3_main = UpsampleBlock1(base_channels, scale_factor=2)
#         self.up3_side = UpsampleBlock1(base_channels, scale_factor=2)
#         self.fuse3 = WarpFusionBlock(base_channels)
#         self.res3 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # --- Output ---
#         self.tail = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

#     def forward(self, c_latent, z_side, flow_field):
#         """
#         c_latent: [B, 4, 32, 32] 主图 Latent
#         z_side:   [B, 4, 32, 32] 边图 Raw Latent (注意：传入原始未 Warp 的)
#         flow_field: [B, 2, 32, 32] 或其他尺寸 (CHW格式)
#         """
#         # 确保 Flow 是 CHW 格式
#         if flow_field.shape[-1] == 2: 
#             flow_field = flow_field.permute(0, 3, 1, 2)

#         # 0. 初始特征提取
#         x = self.head_main(c_latent) # [B, 128, 32, 32]
#         s = self.head_side(z_side)   # [B, 128, 32, 32]
        
#         # --- Level 0 (32x32) [Attention] ---
#         # Flow 不需要 Resize，直接用 (假设 LGAM 输出也是 32)
#         if flow_field.shape[-1] != x.shape[-1]:
#             curr_flow = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
#         else:
#             curr_flow = flow_field
            
#         x = self.fuse0(x, s, curr_flow) # Attention 融合
#         x = self.res0(x)
        
#         # --- Level 1 (32 -> 64) [Attention] ---
#         x = self.up1_main(x)
#         s = self.up1_side(s)
#         curr_flow = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
#         x = self.fuse1(x, s, curr_flow) # Attention 融合
#         x = self.res1(x)
        
#         # --- Level 2 (64 -> 128) [Warp + Concat] ---
#         x = self.up2_main(x)
#         s = self.up2_side(s)
#         curr_flow = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
#         x = self.fuse2(x, s, curr_flow) # Warp 融合
#         x = self.res2(x)
        
#         # --- Level 3 (128 -> 256) [Warp + Concat] ---
#         x = self.up3_main(x)
#         s = self.up3_side(s)
#         curr_flow = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
#         x = self.fuse3(x, s, curr_flow) # Warp 融合
#         x = self.res3(x)
        
#         # --- Output ---
#         out = self.tail(x)
#         return out
#-----------------------------------attn+warp--------------------------------------------------------------------------------------





#----------------------------------------------------------------use attention---------------------------------------------------
# -------------------------------------------------------------------------
# 2. 几何矫正注意力组件 (Positional Encoding & Attention)
# -------------------------------------------------------------------------
# class SinePositionalEncoding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.cached_grid = None # 缓存
        
#     def forward(self, x):
#         B, C, H, W = x.shape
#         device = x.device
        
#         # 如果尺寸变了，或者还没缓存，才重新计算
#         if self.cached_grid is None or self.cached_grid.shape[1:3] != (H, W):
#             y_pos = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(1, H, W)
#             x_pos = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(1, H, W)
#             self.cached_grid = torch.stack([x_pos, y_pos], dim=-1) # [1, H, W, 2]
            
#         # 扩展 Batch 维度 (零开销)
#         return self.cached_grid.expand(B, -1, -1, -1)

# class GeometryCorrectedAttention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim_head = dim // num_heads
#         self.scale = self.dim_head ** -0.5
        
#         self.to_q = nn.Conv2d(dim, dim, 1)
#         self.to_k = nn.Conv2d(dim, dim, 1)
#         self.to_v = nn.Conv2d(dim, dim, 1)
#         self.to_out = nn.Conv2d(dim, dim, 1)
        
#         # 简化位置编码网络
#         self.pos_map = nn.Sequential(
#             nn.Linear(2, dim // 2),
#             nn.SiLU(),
#             nn.Linear(dim // 2, dim),
#         )
#         self.pe_generator = SinePositionalEncoding(dim)

#     def forward(self, x, side, flow):
#         B, C, H, W = x.shape
        
#         # 1. 缓存/生成基础网格 (B, H, W, 2)
#         base_grid = self.pe_generator(x)
        
#         # 2. Key 位置矫正
#         flow_permute = flow.permute(0, 2, 3, 1)
#         grid_k = base_grid + flow_permute
        
#         # 3. 计算 PE
#         combined_grid = torch.cat([base_grid, grid_k], dim=0) # (2B, H, W, 2)
#         combined_pe = self.pos_map(combined_grid).permute(0, 3, 1, 2) # (2B, C, H, W)
#         pe_q, pe_k = torch.chunk(combined_pe, 2, dim=0)
        
#         # 4. 投影
#         q = self.to_q(x + pe_q)
#         k = self.to_k(side + pe_k)
#         v = self.to_v(side)
        
#         # 5. Flash Attention 加速
#         # Reshape: [Batch, Heads, SeqLen, Dim]
#         q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
#         k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
#         v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        
#         # ==================== 🔴 核心修复 ====================
#         # Flash Attention 强制要求输入是内存连续的
#         q = q.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()
#         # ====================================================

#         # 使用 PyTorch 内置的 Flash Attention
#         out = F.scaled_dot_product_attention(q, k, v)
        
#         # Reshape 回去
#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        
#         return self.to_out(out) + x

# # -------------------------------------------------------------------------
# # 3. 几何高保真解码器 (Geometric Fidelity Decoder) - 主类
# # -------------------------------------------------------------------------
# class GeometricFidelityDecoder(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, base_channels=128):
#         super().__init__()
        
#         # -----------------------------------------------------------
#         # A. 主干特征提取 (Main Branch)
#         # -----------------------------------------------------------
#         # 1. 初始投影 (32x32)
#         self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
#         # 2. 初始层融合 (32x32)
#         self.attn0 = GeometryCorrectedAttention(base_channels)
#         self.res0 = ResBlock1(base_channels)
        
#         # 3. 逐级上采样融合
#         # Level 1: 32 -> 64
#         self.up1 = UpsampleBlock1(base_channels, scale_factor=2)
#         self.attn1 = GeometryCorrectedAttention(base_channels)
#         self.res1 = ResBlock1(base_channels) # 可以在attn后加ResBlock进一步提炼
        
#         # Level 2: 64 -> 128
#         self.up2 = UpsampleBlock1(base_channels, scale_factor=2)
#         self.attn2 = GeometryCorrectedAttention(base_channels)
#         self.res2 = ResBlock1(base_channels)
        
#         # Level 3: 128 -> 256
#         self.up3 = UpsampleBlock1(base_channels, scale_factor=2)
#         self.attn3 = GeometryCorrectedAttention(base_channels)
#         self.res3 = ResBlock1(base_channels)
        
#         # 4. 输出层 (256x256)
#         self.tail = nn.Conv2d(base_channels, out_channels, 3, 1, 1)


#         # -----------------------------------------------------------
#         # B. 边信息特征提取 (Side Branch)
#         # -----------------------------------------------------------
#         # 需要把 warped_z_side 也同步上采样，提供给每一层的 Attention 作为 Key/Value
#         self.side_head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
#         self.side_up1 = UpsampleBlock1(base_channels, scale_factor=2)
#         self.side_up2 = UpsampleBlock1(base_channels, scale_factor=2)
#         self.side_up3 = UpsampleBlock1(base_channels, scale_factor=2)

#     def forward(self, c_latent, warped_z_side, flow_field):
#         """
#         c_latent:      [B, 4, 32, 32]
#         warped_z_side: [B, 4, 32, 32]
#         flow_field:    [B, 2, H, W] (原始分辨率或其他分辨率)
#         """
#         # print(flow_field.shape) #torch.Size([32, 64, 2])
#         if flow_field.shape[-1] == 2:
#             flow_field = flow_field.permute(0, 3, 1, 2)
#         # --- 0. 初始特征提取 (32x32) ---
#         x = self.head(c_latent)           # Main Feature
#         side = self.side_head(warped_z_side) # Side Feature
        
#         # 处理 Flow (32x32)
#         if flow_field.shape[-1] != x.shape[-1]:
#             flow_32 = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
#         else:
#             flow_32 = flow_field
            
#         # Layer 0 Fusion (32x32)
#         x = self.attn0(x, side, flow_32)
#         x = self.res0(x)

#         # --- 1. 第一级上采样 (32 -> 64) ---
#         x = self.up1(x)         # Main Up
#         side = self.side_up1(side) # Side Up
        
#         # Resize Flow to 64x64
#         flow_64 = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
#         # Fusion
#         x = self.attn1(x, side, flow_64)
#         x = self.res1(x)

#         # --- 2. 第二级上采样 (64 -> 128) ---
#         x = self.up2(x)
#         side = self.side_up2(side)
        
#         flow_128 = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
#         x = self.attn2(x, side, flow_128)
#         x = self.res2(x)

#         # --- 3. 第三级上采样 (128 -> 256) ---
#         x = self.up3(x)
#         side = self.side_up3(side)
        
#         flow_256 = F.interpolate(flow_field, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
#         x = self.attn3(x, side, flow_256)
#         x = self.res3(x)

#         # --- 4. 输出 ---
#         out = self.tail(x)
#         return out
#-------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------initially direct upsampling--------------------------------------
# class FidelityDecoder(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, base_channels=128, num_res_blocks=6):
#         super().__init__()
        
#         # 1. 头部投影: Latent(4) -> Feature(128)
#         self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
#         # 2. 身体: 堆叠残差块 (Deep Feature Extraction)
#         # num_res_blocks 越多，非线性映射能力越强，PSNR 越高
#         self.body = nn.Sequential(*[
#             ResBlock1(base_channels) for _ in range(num_res_blocks)
#         ])
        
#         # 3. 尾部后处理: 残差连接 (Global Residual Learning)
#         self.conv_tail = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        
#         # 4. 上采样阶段: 3次 2x 上采样 (32 -> 64 -> 128 -> 256)
#         self.upsampler = nn.Sequential(
#             UpsampleBlock1(base_channels, scale_factor=2),
#             UpsampleBlock1(base_channels, scale_factor=2),
#             UpsampleBlock1(base_channels, scale_factor=2),
#         )
        
#         # 5. 输出层: Feature -> RGB
#         self.final = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

#     def forward(self, z):
#         # z: [B, 4, 32, 32]
        
#         # 提取特征
#         x = self.head(z)
        
#         # 残差学习
#         res = self.body(x)
#         res = self.conv_tail(res)
#         x = x + res  # Global Residual Connection
        
#         # 上采样到 256x256
#         x = self.upsampler(x)
        
#         # 输出 RGB
#         out = self.final(x)
        
#         return out
#----------------------------------------------------------initial direct upsampling--------------------------------------------------------------


# #-----------------------------------------------------warped concat--------------------------------------------------------------------------------
# class WarpFusionBlock(nn.Module):
#     """
#     1. Resize Flow
#     2. Warp Side Feature
#     3. Concat Main + Warped Side
#     4. Fusion Conv
#     """
#     def __init__(self, channels):
#         super().__init__()
#         # 融合层: 输入通道翻倍(Main+Side)，输出回原通道
#         self.fusion = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 3, 1, 1),
#             nn.GroupNorm(32, channels),
#             nn.SiLU(),
#             nn.Conv2d(channels, channels, 3, 1, 1) 
#         )

#     def get_grid(self, B, H, W, device):
#         # 生成归一化网格 [-1, 1]
#         y_pos = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(B, H, W)
#         x_pos = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(B, H, W)
#         grid = torch.stack([x_pos, y_pos], dim=-1)
#         return grid

#     def forward(self, x, side, flow_field):
#         """
#         x: Main Feature [B, C, H, W]
#         side: Side Feature [B, C, H, W]
#         flow_field: Base Flow [B, 2, H_base, W_base] OR [B, H_base, W_base, 2]
#         """
#         B, C, H, W = x.shape
        
#         # 1. 维度自动纠正 (关键修复步骤)
#         # 我们需要 [B, 2, H, W] 来做 interpolate
#         # 如果输入是 [B, H, W, 2]，最后一维是 2，说明是 HWC 格式，需要转置
#         if flow_field.shape[-1] == 2:
#             flow_field = flow_field.permute(0, 3, 1, 2)
            
#         # 此时 flow_field 必定是 [B, 2, H_in, W_in]
        
#         # 2. 插值 (Resize Flow)
#         if flow_field.shape[-2:] != (H, W):
#             flow = F.interpolate(flow_field, size=(H, W), mode='bilinear', align_corners=False)
#         else:
#             flow = flow_field
            
#         # 3. 转回 [B, H, W, 2] 用于 grid_sample
#         flow = flow.permute(0, 2, 3, 1)
        
#         # 4. 生成 Grid 并 Warp
#         base_grid = self.get_grid(B, H, W, x.device)
#         sample_grid = base_grid + flow
        
#         warped_side = F.grid_sample(side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
#         # 5. 拼接与融合
#         combined = torch.cat([x, warped_side], dim=1)
#         out = self.fusion(combined)
        
#         # 残差连接
#         return x + out

# class GeometricFidelityDecoder(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, base_channels=128, num_res_blocks=2):
#         super().__init__()
        
#         # --- Head ---
#         self.head_main = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
#         self.head_side = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
#         # --- Level 0 (32x32) ---
#         self.fuse0 = WarpFusionBlock(base_channels)
#         self.res0 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # --- Level 1 (32 -> 64) ---
#         self.up1_main = UpsampleBlock1(base_channels, scale_factor=2)
#         self.up1_side = UpsampleBlock1(base_channels, scale_factor=2)
#         self.fuse1 = WarpFusionBlock(base_channels)
#         self.res1 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # --- Level 2 (64 -> 128) ---
#         self.up2_main = UpsampleBlock1(base_channels, scale_factor=2)
#         self.up2_side = UpsampleBlock1(base_channels, scale_factor=2)
#         self.fuse2 = WarpFusionBlock(base_channels)
#         self.res2 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # --- Level 3 (128 -> 256) ---
#         self.up3_main = UpsampleBlock1(base_channels, scale_factor=2)
#         self.up3_side = UpsampleBlock1(base_channels, scale_factor=2)
#         self.fuse3 = WarpFusionBlock(base_channels)
#         self.res3 = nn.Sequential(*[ResBlock1(base_channels) for _ in range(num_res_blocks)])
        
#         # --- Tail ---
#         self.tail = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

#     def forward(self, c_latent, z_side, flow_field):
#         """
#         c_latent: 主图 Latent [B, 4, 32, 32]
#         z_side:   边图 Latent [B, 4, 32, 32] (原始未Warp)
#         flow_field: 流场 [B, 2, 32, 32]
#         """
#         # 0. 初始投影
#         x = self.head_main(c_latent)
#         s = self.head_side(z_side)
        
#         # ---------------- Level 0 (32x32) ----------------
#         x = self.fuse0(x, s, flow_field)
#         x = self.res0(x)
        
#         # ---------------- Level 1 (64x64) ----------------
#         x = self.up1_main(x)
#         s = self.up1_side(s) # Side 也上采样
#         x = self.fuse1(x, s, flow_field) # flow 会在内部自动 resize 到 64
#         x = self.res1(x)
        
#         # ---------------- Level 2 (128x128) ----------------
#         x = self.up2_main(x)
#         s = self.up2_side(s)
#         x = self.fuse2(x, s, flow_field)
#         x = self.res2(x)
        
#         # ---------------- Level 3 (256x256) ----------------
#         x = self.up3_main(x)
#         s = self.up3_side(s)
#         x = self.fuse3(x, s, flow_field)
#         x = self.res3(x)
        
#         # Output
#         out = self.tail(x)
#         return out
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------warped concat + align --------------------------------------------------------------------------------
class WarpFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GroupNorm(16, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1) 
        )

    def get_grid(self, B, H, W, device):
        y_pos = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(B, H, W)
        x_pos = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(B, H, W)
        grid = torch.stack([x_pos, y_pos], dim=-1)
        return grid

    def forward(self, x, side, flow_field):
        """
        x: Main Feature [B, C, H, W]
        side: Side Feature [B, C, H, W] (如果尺寸不匹配，内部会自动Resize)
        flow_field: Base Flow [B, 2, H_base, W_base] (CHW)
        """
        B, C, H, W = x.shape
        
        # 1. 准备 Flow
        # 如果 flow 分辨率不对，插值到当前分辨率
        if flow_field.shape[-2:] != (H, W):
            flow = F.interpolate(flow_field, size=(H, W), mode='bilinear', align_corners=False)
        else:
            flow = flow_field
        
        # [B, 2, H, W] -> [B, H, W, 2]
        flow_permute = flow.permute(0, 2, 3, 1)
        
        # 2. 准备 Side Feature
        # 如果 side 分辨率不对（因为z_side是32，当前可能是64/128），插值
        if side.shape[-2:] != (H, W):
            side_resized = F.interpolate(side, size=(H, W), mode='bilinear', align_corners=False)
        else:
            side_resized = side
            
        # 3. Warp
        base_grid = self.get_grid(B, H, W, x.device)
        sample_grid = base_grid + flow_permute
        
        warped_side = F.grid_sample(side_resized, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # 4. 融合
        combined = torch.cat([x, warped_side], dim=1)
        out = self.fusion(combined)
        
        return x + out

class GeometricFidelityDecoder(nn.Module):
    def __init__(self, in_channels=4, base_channels=256):
        super().__init__()
        
        # --- Entry ---
        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # --- Encoder (Up) ---
        # Level 0: 32x32
        self.fuse0 = WarpFusionBlock(base_channels)
        self.res0 = ResBlock1(base_channels)
        
        # Level 1: 32 -> 64
        self.up1 = UpsampleBlock1(base_channels, scale_factor=2)
        self.fuse1 = WarpFusionBlock(base_channels)
        self.res1 = ResBlock1(base_channels)
        
        # Level 2: 64 -> 128 (Bottleneck)
        self.up2 = UpsampleBlock1(base_channels, scale_factor=2)
        self.fuse2 = WarpFusionBlock(base_channels)
        self.res2 = nn.Sequential(
            ResBlock1(base_channels),
            ResBlock1(base_channels)
        )
        
        # --- Decoder (Down) ---
        # Level 1: 128 -> 64
        # 使用 stride=2 的卷积进行下采样
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, 2, 1) 
        self.fuse_skip1 = nn.Conv2d(base_channels*2, base_channels, 1) # 融合 Skip Connection
        self.res_d1 = ResBlock1(base_channels)
        
        # Level 0: 64 -> 32
        self.down0 = nn.Conv2d(base_channels, base_channels, 3, 2, 1)
        self.fuse_skip0 = nn.Conv2d(base_channels*2, base_channels, 1)
        self.res_d0 = ResBlock1(base_channels)
        
        # --- Exit ---
        # self.tail = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        
        # # 零初始化最后一层，保证初始输出为0（Identity Mapping）
        # nn.init.zeros_(self.tail.weight)
        # nn.init.zeros_(self.tail.bias)
        self.pixel_head = nn.Sequential(
             # 32 -> 64
             nn.Conv2d(256, 512, 3, 1, 1), nn.PixelShuffle(2), nn.SiLU(),
             # 64 -> 128
             nn.Conv2d(128, 512, 3, 1, 1), nn.PixelShuffle(2), nn.SiLU(),
             # 128 -> 256
             nn.Conv2d(128, 512, 3, 1, 1), nn.PixelShuffle(2), nn.SiLU(),
             # Out RGB
             nn.Conv2d(128, 3, 3, 1, 1)
        )
    def forward(self, c_latent, z_side, flow_field):
        """
        c_latent: [B, 4, 32, 32]
        z_side:   [B, 4, 32, 32] (Raw)
        flow_field: [B, 2, 32, 32] (CHW)
        """
        # 1. 维度检查
        if flow_field.shape[-1] == 2:
            flow_field = flow_field.permute(0, 3, 1, 2)
            
        # 2. 特征提取
        x = self.head(c_latent)           # [B, 128, 32, 32]
        side_feat = self.head(z_side)     # [B, 128, 32, 32] 提取边图特征
        
        # === Encoder Path (Upsampling) ===
        
        # 32x32
        x = self.fuse0(x, side_feat, flow_field)
        x0 = self.res0(x) # Skip Connection 0
        
        # 64x64
        x = self.up1(x0)
        x = self.fuse1(x, side_feat, flow_field) # 内部会自动 resize side 和 flow
        x1 = self.res1(x) # Skip Connection 1
        
        # 128x128 (Bottleneck)
        x = self.up2(x1)
        x = self.fuse2(x, side_feat, flow_field)
        x = self.res2(x)
        
        # === Decoder Path (Downsampling) ===
        
        # 128 -> 64
        x = self.down1(x)
        x = torch.cat([x, x1], dim=1) # Skip Connection
        x = self.fuse_skip1(x)
        x = self.res_d1(x)
        
        # 64 -> 32
        x = self.down0(x)
        x = torch.cat([x, x0], dim=1) # Skip Connection
        x = self.fuse_skip0(x)
        x = self.res_d0(x)
        
        # === Output ===
        # out = self.tail(x)
        
        out = self.pixel_head(x)
        
        # 全局残差连接：输入 + 修正量
        return out
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# class MultiReferenceFusion(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         # 计算权重：输入 [主图特征, 参考图特征] -> 输出 0~1 的权重图
#         self.weight_net = nn.Sequential(
#             nn.Conv2d(channels * 2, channels // 2, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(channels // 2, 1, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, main_feat, ref_feats_stack):
#         """
#         main_feat: [B, C, H, W]
#         ref_feats_stack: [B, N, C, H, W] (N张对齐后的参考图)
#         """
#         B, N, C, H, W = ref_feats_stack.shape
#         main_feat_exp = main_feat.unsqueeze(1).expand(-1, N, -1, -1, -1)
        
#         # 拼接主图与所有参考图计算权重
#         # 转换维度为 [B*N, C*2, H, W]
#         combined = torch.cat([main_feat_exp, ref_feats_stack], dim=2).view(B*N, C*2, H, W)
#         weights = self.weight_net(combined).view(B, N, 1, H, W) # [B, N, 1, H, W]
        
#         # 归一化权重 (Softmax)
#         weights = torch.softmax(weights, dim=1)
        
#         # 加权融合
#         fused = torch.sum(ref_feats_stack * weights, dim=1) # [B, C, H, W]
#         return fused, weights
    # def __init__(self, channels):
    #     super().__init__()
    #     self.channels = channels
        
    #     # 1. 简单的投影，不搞复杂的聚类
    #     self.q_proj = nn.Conv2d(channels, channels, 1)
    #     self.k_proj = nn.Conv2d(channels, channels, 1)
    #     self.v_proj = nn.Conv2d(channels, channels, 1)
        
    #     # 2. 门控层：决定边信息的通过率
    #     self.gate = nn.Sequential(
    #         nn.Conv2d(channels * 2, channels, 3, 1, 1),
    #         nn.SiLU(),
    #         nn.Conv2d(channels, 1, 3, 1, 1),
    #         nn.Sigmoid()
    #     )

    # def forward(self, f_main, f_refs_stack, return_vis=False):
    #     B, K, C, H, W = f_refs_stack.shape
        
    #     # --- 1. 核心逻辑：利用主图引导，从 K 张参考图中“拾取”最有用的信息 ---
    #     # f_main: [B, C, H, W] -> Query
    #     # f_refs: [B, K, C, H, W] -> Keys & Values
        
    #     q = self.q_proj(f_main).view(B, C, -1).permute(0, 2, 1) # [B, HW, C]
    #     k = self.k_proj(f_refs_stack.view(-1, C, H, W)).view(B, K, C, -1).permute(0, 1, 3, 2) # [B, K, HW, C]
    #     v = self.v_proj(f_refs_stack.view(-1, C, H, W)).view(B, K, C, -1).permute(0, 1, 3, 2) # [B, K, HW, C]

    #     # 2. 计算像素级相似度：主图每个点对比所有图的对应点
    #     # 维度: [B, HW, K] (这里只对比对应空间位置，大大降低计算量)
    #     # 我们采用更简单的局部注意力或 Pool 后的注意力
    #     attn = torch.softmax(torch.mean(q.unsqueeze(1) * k, dim=-1), dim=1) # [B, K, HW]
        
    #     # 3. 融合：[B, C, HW]
    #     f_side = torch.sum(v * attn.unsqueeze(-1), dim=1).permute(0, 2, 1).view(B, C, H, W)
        
    #     # 4. 门控注入 (Residual)
    #     g = self.gate(torch.cat([f_main, f_side], dim=1))
    #     f_fused = f_main + g * f_side
        
    #     # 为了可视化，返回 attn 即可 (这是真正的 resolved attention)
    #     return f_fused, attn.view(B, K, H, W), None, None, None
    
class LGAM(nn.Module):
    """
    Learnable Geometric Alignment Module
    利用主/边视角的特征差异，自适应学习几何参数，结合边视角深度图，
    同时对边视角的特征(feat)和潜变量(latent)进行对齐。
    """
    def __init__(self, feature_dim=256, latent_dim=4):
        super().__init__()
        
        # -------------------------------------------------------
        # 1. 全局参数回归器 (Global Parameter Regressor)
        # -------------------------------------------------------
        # 输入：Concat(feat_main, feat_side) -> [B, 512, 32, 32]
        # 输出：alpha (scale), beta (shift) -> [B, 2]
        self.global_param_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 128, kernel_size=3, stride=2, padding=1), # 32->16
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), # 16->8
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1), # [B, 64, 1, 1]
            nn.Flatten(),
            nn.Linear(64, 2)         # [alpha, beta]
        )
        # 初始化：让初始状态接近 identity (alpha=0, beta=0) 或者合理的初始值
        # nn.init.zeros_(self.global_param_net[-1].weight)
        # nn.init.zeros_(self.global_param_net[-1].bias)

        # -------------------------------------------------------
        # 2. 局部残差流预测器 (Local Residual Flow Predictor)
        # -------------------------------------------------------
        # 输入：feat_main, feat_side, coarse_flow, depth_feature
        # 输出：delta_flow [B, 2, 32, 32]
        self.res_flow_net = nn.Sequential(
            # 输入通道：256(main) + 256(side) + 1(coarse_x) + 1(depth) = 514
            nn.Conv2d(feature_dim * 2 + 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 2, kernel_size=3, padding=1) # 输出 x, y 方向的微调
        )
        # 初始化为0，保证初始训练稳定
        # nn.init.xavier_uniform_(self.global_param_net[-1].weight)
        # nn.init.constant_(self.global_param_net[-1].bias, 0.0)

    def get_grid(self, B, H, W, device):
        # 创建归一化网格 [-1, 1]
        xx = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], dim=1) # [B, 2, H, W]
        return grid

    def forward(self, depth_side_raw, feat_main, feat_side, z_main, z_side, return_vis=False):
        """
        Args:
            depth_side_raw: [B, 1, H_orig, W_orig] 原始深度图
            feat_main:      [B, 256, 32, 32]       主特征 (Ground Truth 参考)
            feat_side:      [B, 256, 32, 32]       边特征 (Source)
            z_main:         [B, 4, 32, 32]         主潜变量 (仅用于对其，不参与计算)
            z_side:         [B, 4, 32, 32]         边潜变量 (Source)
        
        Returns:
            warped_feat_side: [B, 256, 32, 32]
            warped_z_side:    [B, 4, 32, 32]
        """
        B, C, H, W = feat_main.shape # 32, 32
        device = feat_main.device

        # ---------------------------------------------------
        # 1. 深度图预处理
        # ---------------------------------------------------
        # 下采样深度图到特征尺寸
        depth_small = F.interpolate(depth_side_raw, size=(H, W), mode='bilinear', align_corners=False)
        
        # 计算逆深度 (Inverse Depth)，归一化到 [0, 1] 方便网络学习
        inv_depth = 1.0 / (depth_small + 1e-6)
        inv_depth_norm = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min() + 1e-6)

        params = self.global_param_net(torch.cat([feat_main.detach(), feat_side], dim=1))
        # params = self.global_param_net(torch.cat([feat_main, feat_side], dim=1))
        
        # alpha = params[:, 0].view(B, 1, 1, 1) # 视差缩放系数 (f * B)
        alpha = F.softplus(params[:, 0]).view(B, 1, 1, 1) + 0.05



        beta  = params[:, 1].view(B, 1, 1, 1) # 全局偏移 (Shift)
        # beta = torch.tanh(params[:, 1]).view(B, 1, 1, 1)

        # ---------------------------------------------------
        # 3. 构建粗糙物理流 (Coarse Physical Flow)
        # ---------------------------------------------------
        # 物理公式：Disp = alpha * (1/Z) + beta
        # 假设主要是水平位移 (Stereo)
        flow_coarse_x = alpha * inv_depth_norm + beta
        
        # ---------------------------------------------------
        # 4. 预测残差流 (Residual Flow)
        # ---------------------------------------------------
        # 拼接所有信息：特征 + 物理先验 + 深度形状
        refine_input = torch.cat([feat_main.detach(), feat_side, flow_coarse_x, inv_depth_norm], dim=1)
        # refine_input = torch.cat([feat_main, feat_side, flow_coarse_x, inv_depth_norm], dim=1)
        delta_flow = self.res_flow_net(refine_input) # [B, 2, H, W]

        # ---------------------------------------------------
        # 5. 合成最终流场 (Final Flow)
        # ---------------------------------------------------
        # 基础流 (X轴物理流, Y轴为0) + 残差流
        final_flow_x = flow_coarse_x + delta_flow[:, 0:1, ...]
        final_flow_y = delta_flow[:, 1:2, ...] # 允许少量的垂直修正
        
        # 拼接并调整维度为 [B, H, W, 2] 用于 grid_sample
        # 注意：这里的 flow 是归一化坐标下的偏移量 (因为 grid 是 -1 到 1)
        # 如果网络输出比较大，可能需要 tanh 或者 scale 限制，这里假设网络能自己学
        flow_field = torch.cat([final_flow_x, final_flow_y], dim=1).permute(0, 2, 3, 1)

        # 生成采样网格
        base_grid = self.get_grid(B, H, W, device).permute(0, 2, 3, 1)
        sample_grid = base_grid + flow_field

        # ---------------------------------------------------
        # 6. 执行 Warp
        # ---------------------------------------------------
        # 对 Feature 进行 Warp
        warped_feat_side = F.grid_sample(feat_side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # 对 Latent 进行 Warp (使用完全相同的 grid)
        warped_z_side = F.grid_sample(z_side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        vis_data = None
        if return_vis:
            vis_data = {
                "inv_depth": inv_depth_norm.detach(),
                "flow_x": final_flow_x.detach(), # 只看X方向，因为双目主要是水平位移
                "alpha": alpha.detach().mean(),  # 取Batch平均值用于记录
                "beta": beta.detach().mean()
            }

        return warped_feat_side, warped_z_side, vis_data, flow_field
        # return warped_feat_side, warped_z_side

import torch
import torch.nn as nn
import torch.nn.functional as F

# class LGAM(nn.Module):
#     """
#     Generalized Learnable Geometric Alignment Module (G-LGAM)
#     1. 支持非水平视角的几何校正 (x, y 全向参数回归)
#     2. 引入对齐置信度掩码 (Confidence Mask)，自动过滤无关/对不齐的信息
#     3. 增强型残差流预测
#     """
#     def __init__(self, feature_dim=256, latent_dim=4):
#         super().__init__()
        
#         # -------------------------------------------------------
#         # 1. 全局几何参数回归器 (支持 X, Y 双向物理参数)
#         # -------------------------------------------------------
#         # 输入：Concat(feat_main, feat_side)
#         # 输出：[alpha_x, beta_x, alpha_y, beta_y]
#         self.global_param_net = nn.Sequential(
#             nn.Conv2d(feature_dim * 2, 128, kernel_size=3, stride=2, padding=1),
#             nn.GroupNorm(8, 128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(64, 4) # 回归四个全局参数
#         )

#         # -------------------------------------------------------
#         # 2. 局部残差流 & 置信度掩码预测器
#         # -------------------------------------------------------
#         # 输入：feat_main, feat_side, coarse_flow, inv_depth
#         # 输出：[delta_x, delta_y, mask]
#         self.refine_net = nn.Sequential(
#             nn.Conv2d(feature_dim * 2 + 3, 128, kernel_size=3, padding=1), # +3 = coarse_x, coarse_y, depth
#             nn.GroupNorm(8, 128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(64, 3, kernel_size=3, padding=1) # 输出 2(flow) + 1(mask)
#         )

#     def get_grid(self, B, H, W, device):
#         xx = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
#         yy = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)
#         return torch.cat([xx, yy], dim=1)

#     def forward(self, depth_side_raw, feat_main, feat_side, z_main, z_side, return_vis=False):
#         B, C, H, W = feat_main.shape
#         device = feat_main.device

#         # ---------------------------------------------------
#         # 1. 深度图预处理
#         # ---------------------------------------------------
#         depth_small = F.interpolate(depth_side_raw, size=(H, W), mode='bilinear', align_corners=False)
#         inv_depth = 1.0 / (depth_small + 1e-6)
#         inv_depth_norm = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min() + 1e-6)

#         # ---------------------------------------------------
#         # 2. 回归全局仿射诱导参数
#         # ---------------------------------------------------
#         params = self.global_param_net(torch.cat([feat_main.detach(), feat_side], dim=1))
        
#         # 使用 Softplus 保证 alpha (视差增益) 为正
#         alpha_x = (F.softplus(params[:, 0]) + 0.1).view(B, 1, 1, 1)
#         beta_x  = params[:, 1].view(B, 1, 1, 1)
        
#         # Y 方向参数（通常较小，但在非对齐检索中至关重要）
#         alpha_y = params[:, 2].view(B, 1, 1, 1) * 0.1 
#         beta_y  = params[:, 3].view(B, 1, 1, 1) * 0.1

#         # 构建粗糙物理流 (Coarse Flow)
#         flow_coarse_x = alpha_x * inv_depth_norm + beta_x
#         flow_coarse_y = alpha_y * inv_depth_norm + beta_y

#         # ---------------------------------------------------
#         # 3. 预测局部残差流 & 置信度掩码
#         # ---------------------------------------------------
#         refine_input = torch.cat([
#             feat_main.detach(), 
#             feat_side, 
#             flow_coarse_x, 
#             flow_coarse_y, 
#             inv_depth_norm
#         ], dim=1)
        
#         refine_out = self.refine_net(refine_input)
#         delta_flow = refine_out[:, :2, ...]   # 局部微调
#         # 对齐置信度掩码：0 表示放弃该像素，1 表示完全采用
#         conf_mask = torch.sigmoid(refine_out[:, 2:3, ...]) 

#         # ---------------------------------------------------
#         # 4. 合成最终采样网格
#         # ---------------------------------------------------
#         final_flow_x = flow_coarse_x + delta_flow[:, 0:1, ...]
#         final_flow_y = flow_coarse_y + delta_flow[:, 1:2, ...]
        
#         flow_field = torch.cat([final_flow_x, final_flow_y], dim=1).permute(0, 2, 3, 1)
#         base_grid = self.get_grid(B, H, W, device).permute(0, 2, 3, 1)
#         sample_grid = base_grid + flow_field

#         # ---------------------------------------------------
#         # 5. 执行 Warp 并应用掩码
#         # ---------------------------------------------------
#         # 对 Feature 进行对齐并过滤噪声
#         warped_feat_side = F.grid_sample(feat_side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
#         warped_feat_side = warped_feat_side * conf_mask # 语义筛选

#         # 对 Latent 进行对齐并过滤
#         warped_z_side = F.grid_sample(z_side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
#         warped_z_side = warped_z_side * conf_mask

#         # ---------------------------------------------------
#         # 6. 整理可视化数据
#         # ---------------------------------------------------
#         vis_data = None
#         if return_vis:
#             vis_data = {
#                 "inv_depth": inv_depth_norm.detach(),
#                 "flow_x": final_flow_x.detach(),
#                 "flow_y": final_flow_y.detach(),
#                 "conf_mask": conf_mask.detach(), # 极其重要的筛选可视化
#                 "alpha": alpha_x.detach().mean(),
#                 "beta": beta_x.detach().mean()
#             }

#         return warped_feat_side, warped_z_side, vis_data, flow_field

class MultiReferenceFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Conv2d(channels, channels // 4, 1)
        self.k_proj = nn.Conv2d(channels, channels // 4, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        num_groups = min(channels, 32)
        if channels % num_groups != 0: num_groups = 1
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU()
        )

    def forward(self, f_main, f_refs_stack):
        """
        f_main: [B, C, H, W]
        f_refs_stack: [B, K, C, H, W]
        """
        B, K, C, H, W = f_refs_stack.shape
        Cp = C // 4
        
        q = self.q_proj(f_main) # [B, Cp, H, W]
        k = self.k_proj(f_refs_stack.view(-1, C, H, W)).view(B, K, Cp, H, W)
        v = self.v_proj(f_refs_stack.view(-1, C, H, W)).view(B, K, C, H, W)

        # 1. 计算原始相似度曲面 (Dot Product Surface)
        # sim: [B, K, H, W]
        sim = torch.sum(q.unsqueeze(1) * k, dim=2) / (Cp ** 0.5)
        
        # 2. Winner-Take-All (WTA) 硬选择
        # 找到每个位置最相似的参考图索引 (即曲面最凸点)
        best_idx = torch.argmax(sim, dim=1) # [B, H, W]
        
        # 构造硬掩码 (One-hot)
        mask_hard = F.one_hot(best_idx, num_classes=K).permute(0, 3, 1, 2).float() # [B, K, H, W]
        
        # 3. 使用 Straight-Through Estimator 保持端到端梯度
        # 前向使用 mask_hard，反向梯度使用 softmax
        attn_soft = torch.softmax(sim, dim=1)
        mask_ste = mask_hard + attn_soft - attn_soft.detach()
        
        # 4. 提取唯一最相似特征并融合
        f_side = torch.sum(v * mask_ste.unsqueeze(2), dim=1) # [B, C, H, W]
        f_fused = self.refine(f_side) + f_main
        # 返回: 融合结果, 最终掩码(attn_weights), 原始相似度(sim_scores)
        return self.refine(f_side), mask_hard, sim
    
class NoiseEstimator(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            use_linear_in_transformer=False,
            control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
            learn_embedding=True,
            control_scale=1.0
    ):
        super().__init__()

        self.learn_embedding = learn_embedding
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.control_scale = control_scale

        ################# start control model variations #################
        base_model = UNetModel(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
        )  # initialise control model from base model
        self.control_model = ControlModule(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
            control_model_ratio=control_model_ratio,
        )  # initialise pretrained model

        ################# end control model variations #################

        self.enc_zero_convs_out = nn.ModuleList([])

        self.middle_block_out = None
        self.middle_block_in = None

        self.dec_zero_convs_out = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        ################# Gather Channel Sizes #################
        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        ################# Build zero convolutions #################
        self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

        self.dec_zero_convs_out.append(
            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(ch_inout_ctr['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
            )
        for i in range(len(ch_inout_ctr['enc'])):
            self.enc_zero_convs_out.append(self.make_zero_conv(
                in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
            )

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list) * self.control_scale)

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def forward(self, x, guide_hint, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.control_model.time_embed(t_emb)
        emb_base = base_model.time_embed(t_emb)
        
        h_base = x.type(base_model.dtype)
        # print("h_base:",h_base.shape) #[4, 4, 32, 32]
        # print("guide_hint:",guide_hint.shape) #[4, 256, 32, 32]
        h_ctr = torch.cat((h_base, guide_hint), dim=1)
        hs_base = []
        hs_ctr = []
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list * self.control_scale)

        ###################### Cross Control  ######################

        # input blocks (encoder)
        for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
            h_base = module_base(h_base, emb_base, context)
            h_ctr = module_ctr(h_ctr, emb, context)
            
            h_base = h_base + next(it_enc_convs_out)(h_ctr, emb) * next(scales)

            hs_base.append(h_base)
            hs_ctr.append(h_ctr)

        # mid blocks (bottleneck)
        h_base = base_model.middle_block(h_base, emb_base, context)
        h_ctr = self.control_model.middle_block(h_ctr, emb, context)

        h_base = h_base + self.middle_block_out(h_ctr, emb) * next(scales)

        # output blocks (decoder)
        for module_base in base_model.output_blocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb) * next(scales)

            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)

        return base_model.out(h_base)
    
    def forward_unconditional(self, x, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb_base = base_model.time_embed(t_emb)

        h_base = x.type(base_model.dtype)
        hs_base = []

        ###################### Cross Control  ######################
        # input blocks (encoder)
        for module_base in base_model.input_blocks:
            h_base = module_base(h_base, emb_base, context)
            hs_base.append(h_base)

        # mid blocks (bottleneck)
        h_base = base_model.middle_block(h_base, emb_base, context)

        # output blocks (decoder)
        for module_base in base_model.output_blocks:
            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)

        return base_model.out(h_base)
    
class ControlModule(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        control_model_ratio=1.0,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        model_channels = int(model_channels * control_model_ratio)
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels+hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size = self._feature_size+ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

def find_denominator(number, start):
    if start >= number:
        return number
    while (start != 0):
        residual = number % start
        if residual == 0:
            return start
        start -= 1

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # if find_denominator(channels, 32) < 32:
    #     print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
    return GroupNorm_leq32(find_denominator(channels, 32), channels)

class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
  
  
  
import numpy as np

class STanHAnnealingStrategy:
    def __init__(self, type="linear", initial_beta=1.0, max_beta=100.0, k=15):
        self.type = type
        self.beta = initial_beta
        self.max_beta = max_beta
        self.k = k # 对应你代码中的 K 因子

    def step(self, gap, epoch=None, loss=None):
        # 对应你代码中的 beta_max_t = beta_max_t_1 + K * Et
        if self.type == "standard":
            # 计算增量
            delta = self.k * gap
            self.beta = min(self.max_beta, self.beta + delta)
        elif self.type == "linear":
            self.beta = min(self.max_beta, self.beta + 0.01)
        
        return self.beta  
    
class RDEIC(LatentDiffusion):

    def __init__(
        self, 
        control_stage_config: Mapping[str, Any], 
        sd_locked: bool,
        is_refine: bool,
        fixed_step: int,
        # synch_control: bool,
        learning_rate: float,
        l_bpp_weight: float,
        l_guide_weight: float,
        used_timesteps: int,
        sync_path: str, 
        synch_control: bool,
        ckpt_path_pre: str,
        preprocess_config: Mapping[str, Any],
        calculate_metrics: Mapping[str, Any],
        train_distortion_decoder_only:bool,
        rate_signal: float,
        n_refs=1, 
        retrieval_index_path=None,
        retrieval_meta_path=None,
        *args, 
        **kwargs
    ) -> "RDEIC":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model = instantiate_from_config(control_stage_config) # model.rdeic.NoiseEstimator
        self.preprocess_model = instantiate_from_config(preprocess_config) # model.compression.Compression
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)
        if ckpt_path_pre is not None:
            self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)

        self.sd_locked = sd_locked
        self.is_refine = is_refine
        self.fixed_step = fixed_step

        self.learning_rate = learning_rate
        self.l_bpp_weight = l_bpp_weight
        self.l_guide_weight = l_guide_weight

        assert used_timesteps <= self.num_timesteps, f'used_timesteps ({used_timesteps}) must be less than or equal to the total number of timesteps ({self.num_timesteps})'
        self.used_timesteps = used_timesteps

        self.calculate_metrics = calculate_metrics
        self.metric_funcs = {}
        for _, opt in calculate_metrics.items(): 
            mopt = opt.copy()
            name = mopt.pop('type', None)
            mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        self.lamba = self.sqrt_recipm1_alphas_cumprod[self.used_timesteps - 1]

        if self.is_refine:
            self.sampler = SpacedSampler(self)
            self.perceptual_loss = LPIPS(pnet_type='vgg')

        self.lgam = LGAM()

        print("Loading Depth Anything V3...")
        self.depth_model = DepthAnything3.from_pretrained("depth_anything_3/DA3NESTED-GIANT-LARGE")
        self.depth_model.eval()
        for param in self.depth_model.parameters():
            param.requires_grad = False

        self.fusion_feature = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256), 
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.fusion_latent = nn.Sequential(
            nn.Conv2d(4 * 2, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, 4, kernel_size=1)      
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(4 * 2, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid() # 限制在 0~1 之间
        )
        self.train_distortion_decoder_only=train_distortion_decoder_only
        self.rate_signal = rate_signal
        # self.distortion_decoder = GeometricFidelityDecoder()
        if self.train_distortion_decoder_only:
            print(">>> MODE: Training Distortion Decoder ONLY. Freezing all other parts.")
            
            # 1. 先冻结所有参数 (包括 inherited from LatentDiffusion 的)
            for param in self.parameters():
                param.requires_grad = False
            
            # 2. 独独解冻 distortion_decoder
            for param in self.preprocess_model.stanh_y.parameters():
                param.requires_grad = True
        # else:
        #     for param in self.distortion_decoder.parameters():
        #         param.requires_grad = False    

        self.use_retrieval = True
        self.n_refs = n_refs
        self.retrieval_index_path = retrieval_index_path
        self.retrieval_meta_path = retrieval_meta_path
        
        # 加载 FAISS 索引
        # self.retrieval_db_path = "retrieval_db"
        # if os.path.exists(self.retrieval_db_path):
        #     self.index = faiss.read_index(os.path.join(self.retrieval_db_path, "gallery.index"))
        #     self.metadata = np.load(os.path.join(self.retrieval_db_path, "metadata.npy"))
        #     print(f"Retrieval DB loaded: {len(self.metadata)} images")

        if self.retrieval_index_path and os.path.exists(self.retrieval_index_path):
            import faiss
            loaded_index = faiss.read_index(self.retrieval_index_path)
            self.metadata = np.load(self.retrieval_meta_path)
            
            if not isinstance(loaded_index, faiss.IndexIDMap):
                print("🔄 正在将普通索引转换为支持实时更新的 IDMap2 格式...")
                dim = loaded_index.d
                ntotal = loaded_index.ntotal
                all_vectors = loaded_index.reconstruct_n(0, ntotal)
                
                new_base = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIDMap2(new_base) # 这里传入的是空的 new_base，所以正确
                
                ids = np.arange(ntotal).astype('int64')
                self.index.add_with_ids(all_vectors, ids)
                print(f"✅ 转换完成！当前库容量: {self.index.ntotal}")
            else:
                self.index = loaded_index
                print(f"✅ 检索库加载成功 (已是 IDMap 格式): {self.index.ntotal} images")

        # --- 2. 初始化融合层参数 (保留) ---
        self.fusion_feature_multi = MultiReferenceFusion(256)
        self.fusion_latent_multi = MultiReferenceFusion(4)
        self.gallery_momentum = 0.99 
        if hasattr(self, 'metadata') and self.metadata is not None:
            # 建立 路径 -> 索引ID 的映射表
            self.path_to_id = {os.path.abspath(path): i for i, path in enumerate(self.metadata)}
            print(f"✅ Path-to-ID mapping established for {len(self.path_to_id)} images.")        
        # self.anneal_z = STanHAnnealingStrategy(type="standard", k=15)
        # self.anneal_y = STanHAnnealingStrategy(type="standard", k=15)
        # self.current_beta_z = 1.0
        # self.current_beta_y = 1.0
        # self.K_factor = 15.0 
        
        # # 使用 register_buffer 保证 beta_max 会被保存在 checkpoint 中
        # # 初始化 beta_max_1 = 1 (Equation 8)
        # self.register_buffer("beta_max_y", torch.ones(1))
        # self.register_buffer("beta_max_z", torch.ones(1))

    # def training_step(self, batch, batch_idx):
    #     # 1. 在 get_input 之前，准备当前的 beta
    #     # 注意：分布式场景通常 y 的量化对质量影响更大
        
    #     # 2. 执行模型前向（通过 get_input 最终调用 preprocess_model）
    #     # 需要修改 get_input 的定义来接收 beta
    #     z, cond = self.get_input(batch, self.first_stage_key, 
    #                              beta_y=self.current_beta_y, 
    #                              beta_z=self.current_beta_z)
        
    #     # 3. 提取 Gap 并更新 Beta (对应你 train_one_epoch 里的逻辑)
    #     if "gaps" in cond:
    #         gap_z, gap_y = cond["gaps"]
    #         self.current_beta_z = self.anneal_z.step(gap_z)
    #         self.current_beta_y = self.anneal_y.step(gap_y)

    #     # 4. 计算损失并返回
    #     loss, loss_dict = self.p_losses(z, cond, ...)
        
    #     # 5. WandB 日志记录 (对应你代码里的 wandb.log)
    #     self.log("stanh/beta_y", self.current_beta_y, prog_bar=True)
    #     self.log("stanh/gap_y", gap_y)
        
    #     return loss




    # def apply_condition_encoder(self, x, y):
    #     c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y = self.preprocess_model(x,y)
    #     return c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y
    
    def apply_condition_encoder(self, x, y):
            c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y = self.preprocess_model(x, y)
            return c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y    

    @torch.no_grad()
    def apply_condition_compress(self, x, stream_path, H, W):
        _, h = self.encode_first_stage(x * 2 - 1)
        h = h * self.scale_factor
        out = self.preprocess_model.compress(h)
        shape = out["shape"]
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, out["strings"])
        size = filesize(stream_path)
        bpp = float(size) * 8 / (H * W)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(self, stream_path):
        with Path(stream_path).open("rb") as f:
            strings, shape = read_body(f)
        c_latent, guide_hint = self.preprocess_model.decompress(strings, shape)
        return c_latent, guide_hint

    @torch.no_grad()
    def save_lgam_debug(self, save_path, vis_data):
        import cv2
        import numpy as np
        import os
        
        VIS_SIZE = (512, 256) # 根据你的输入调整，如果是256x512则设为 (512, 256)
        base_path, ext = os.path.splitext(save_path)
        
        # 获取 Batch Size
        B = vis_data['inv_depth'].shape[0]

        # 改进后的热力图生成函数
        def to_heatmap(tensor_hw, colormap=cv2.COLORMAP_INFERNO):
            arr = tensor_hw.cpu().numpy()
            
            # 使用百分位拉伸对比度
            p_min = np.percentile(arr, 2)
            p_max = np.percentile(arr, 98)
            
            arr = np.clip(arr, p_min, p_max)
            
            if p_max - p_min > 1e-6:
                arr = (arr - p_min) / (p_max - p_min)
            else:
                arr = np.zeros_like(arr)
                
            arr = (arr * 255).astype(np.uint8)
            arr = cv2.resize(arr, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
            return cv2.applyColorMap(arr, colormap)

        # 遍历 Batch 保存
        for i in range(1):
            # 1. Inverse Depth
            img_depth = to_heatmap(vis_data['inv_depth'][i, 0], cv2.COLORMAP_INFERNO)
            
            # 2. Predicted Flow X
            # 使用 JET 色图区分正负或大小差异
            img_flow = to_heatmap(vis_data['flow_x'][i, 0], cv2.COLORMAP_JET)
            
            # 3. 拼接
            canvas = np.hstack([img_depth, img_flow])
            
            # 4. 获取参数
            # 注意：如果 LGAM 内部对 alpha/beta 做了 mean，这里得到的可能是同一个值
            # 如果没做 mean，这里取第 i 个
            if vis_data['alpha'].ndim > 0 and vis_data['alpha'].shape[0] == B:
                a_val = vis_data['alpha'][i].item()
                b_val = vis_data['beta'][i].item()
            else:
                # 兼容 scalar 的情况
                a_val = vis_data['alpha'].item()
                b_val = vis_data['beta'].item()
            
            # 5. 绘制文字
            # 顶部黑色背景条
            # cv2.rectangle(canvas, (0, 0), (VIS_SIZE[0]*2, 40), (0, 0, 0), -1)
            
            info_text = f"Alpha: {a_val:.4f} | Beta: {b_val:.4f}"
            cv2.putText(canvas, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            
            # 底部标签
            h_canvas = canvas.shape[0]
            w_half = canvas.shape[1] // 2
            cv2.putText(canvas, "Inverse Depth", (10, h_canvas - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, "Flow X", (w_half + 10, h_canvas - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 6. 保存：加上 batch 索引
            cv2.imwrite(f"{base_path}_b{i}{ext}", canvas)
    # def save_lgam_debug(self, save_path, vis_data):
    #     import cv2
    #     import numpy as np
        
    #     VIS_SIZE = (256, 256)
        
    #     def to_heatmap(tensor, colormap=cv2.COLORMAP_INFERNO):
    #         # tensor: [1, H, W]
    #         arr = tensor.squeeze().cpu().numpy()
            
    #         # 归一化到 0-255
    #         _min, _max = arr.min(), arr.max()
    #         if _max - _min > 1e-6:
    #             arr = (arr - _min) / (_max - _min)
    #         else:
    #             arr = np.zeros_like(arr)
                
    #         arr = (arr * 255).astype(np.uint8)
    #         arr = cv2.resize(arr, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
    #         return cv2.applyColorMap(arr, colormap)

    #     # 1. 可视化逆深度 (越亮越近)
    #     # 使用 INFERNO 色图 (黑->红->黄)
    #     img_depth = to_heatmap(vis_data['inv_depth'][0], cv2.COLORMAP_INFERNO)
        
    #     # 2. 可视化 Flow X (位移场)
    #     # 使用 JET 色图 (蓝->绿->红)，能看清正负和大小变化
    #     # 理论上 Flow 应该和 Depth 长得很像 (因为 d = alpha * 1/Z + beta)
    #     img_flow = to_heatmap(vis_data['flow_x'][0], cv2.COLORMAP_JET)
        
    #     # 3. 拼接
    #     canvas = np.hstack([img_depth, img_flow])
        
    #     # 4. 在图上打印 Alpha 和 Beta 参数
    #     alpha_val = vis_data['alpha'].item()
    #     beta_val = vis_data['beta'].item()
        
    #     # 绘制文字背景条
    #     cv2.rectangle(canvas, (0, 0), (512, 40), (0, 0, 0), -1)
    #     text = f"Alpha (Scale): {alpha_val:.4f} | Beta (Shift): {beta_val:.4f}"
    #     cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.7, (255, 255, 255), 2)
        
    #     # 添加标题
    #     cv2.putText(canvas, "Inverse Depth", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    #     cv2.putText(canvas, "Predicted Flow X", (266, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #     cv2.imwrite(save_path, canvas)


    @torch.no_grad()
    def save_debug_vis(self, save_path, 
                       z_main, z_warp, z_fused, 
                       f_main, f_warp, f_fused):
        """
        将 Latent 和 Feature 拼图并保存到 save_path
        适配输入尺寸 256(H) x 512(W)
        """
        # 🟢 关键修改：设置统一的可视化尺寸 (Width, Height)
        # 你的原图是 H=256, W=512，所以这里设为 (512, 256)
        VIS_SIZE = (512, 256) 

        # --- A. Latent 转 RGB ---
        def decode_to_img(z):
            # z: [B, 4, 32, 64] (因为输入是256x512, f=8)
            img = self.decode_first_stage(z[0:1]) 
            img = (img.permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5
            img = img.clip(0, 255).astype(np.uint8)[0]
            
            # 强制 Resize 到统一尺寸，防止 VAE 输出有细微偏差
            if (img.shape[1], img.shape[0]) != VIS_SIZE:
                img = cv2.resize(img, VIS_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # RGB -> BGR (OpenCV)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_z_main = decode_to_img(z_main)
        img_z_warp = decode_to_img(z_warp)
        img_z_fused = decode_to_img(z_fused)
        
        # --- B. Feature 转 Heatmap ---
        def feat_to_heatmap(f):
            # f: [B, C, 32, 64]
            heatmap = f[0].mean(dim=0).cpu().numpy()
            
            # 归一化到 [0, 255]
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max - h_min > 1e-6:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap)
            
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # 🟢 关键修改：Resize 到 (512, 256) 以匹配 Latent 图
            heatmap = cv2.resize(heatmap, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
            return cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

        map_f_main = feat_to_heatmap(f_main)
        map_f_warp = feat_to_heatmap(f_warp)
        map_f_fused = feat_to_heatmap(f_fused)
        
        # --- C. 拼图 ---
        # 第一行：Latent (主 | Warp | 融合) -> 宽度 512*3 = 1536
        row1 = np.hstack([img_z_main, img_z_warp, img_z_fused])
        
        # 第二行：Feature (主 | Warp | 融合) -> 宽度 512*3 = 1536
        row2 = np.hstack([map_f_main, map_f_warp, map_f_fused])
        
        # 垂直堆叠
        canvas = np.vstack([row1, row2])
        
        # 保存
        cv2.imwrite(save_path, canvas)



    def _get_depth_maps(self, images_tensor):
            """
            输入: images_tensor [B, 3, H, W] 范围 [-1, 1]
            输出: depth_tensor [B, 1, H, W] 范围 [0, 1] 或 归一化深度
            """
            # 1. 确保深度模型在正确的设备上
            # 注意：DepthModel 内部参数可能在 CPU 或 GPU，这里根据第一个参数判断
            if next(self.depth_model.parameters()).device != images_tensor.device:
                self.depth_model.to(images_tensor.device)

            B, C, H, W = images_tensor.shape
            depth_maps = []

            # 2. 数据转换: Tensor(-1,1) -> Numpy(0,255) uint8 (H, W, 3)
            images_np = ((images_tensor + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            
            # 3. 逐张推理
            with torch.no_grad():
                for i in range(B):
                    img_np = images_np[i]
                    
                    # ========================== 关键修改 ==========================
                    # 必须传入 list，否则 parallel_utils 无法计算 length 报错
                    prediction = self.depth_model.inference([img_np])
                    
                    # 取出第一张图的深度 (inference 返回的是 batch 结果)
                    raw_depth = prediction.depth[0] 
                    # =============================================================
                    
                    # 4. 强制 Resize 回原始尺寸 (H, W)
                    # cv2.resize 接收 (Width, Height)
                    if raw_depth.shape != (H, W):
                        raw_depth = cv2.resize(raw_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                    depth_maps.append(raw_depth)

            # 5. 堆叠并转回 Tensor
            depth_batch = np.stack(depth_maps, axis=0) # [B, H, W]
            depth_tensor = torch.from_numpy(depth_batch).unsqueeze(1).to(images_tensor.device).float() # [B, 1, H, W]
            
            # 6. 归一化到 [0, 1]
            d_min = depth_tensor.flatten(2).min(2)[0].view(B, 1, 1, 1)
            d_max = depth_tensor.flatten(2).max(2)[0].view(B, 1, 1, 1)
            # 加上 1e-6 防止除以 0
            depth_tensor = (depth_tensor - d_min) / (d_max - d_min + 1e-6)

            return depth_tensor
 
    @torch.no_grad()
    def save_depth_vis(self, save_path, depth_tensor):
        """
        保存 Batch 中每一张深度图
        depth_tensor: [B, 1, H, W]
        """
        
        B = depth_tensor.shape[0]
        base_path, ext = os.path.splitext(save_path) # 分离文件名和后缀 (.png)

        # 遍历 Batch
        for i in range(B):
            # 取第 i 张
            d = depth_tensor[i, 0].cpu().numpy()
            
            # 归一化到 0-255
            d_min, d_max = d.min(), d.max()
            if d_max - d_min > 1e-6:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
                
            d_uint8 = (d_norm * 255).astype(np.uint8)
            
            # 伪彩色映射
            d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_VIRIDIS)
            
            # Resize
            VIS_SIZE = (512, 256) 
            d_color = cv2.resize(d_color, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
            
            # 保存：加上 batch 索引后缀，例如 step_00500_depth_b0.png
            cv2.imwrite(f"{base_path}_b{i}{ext}", d_color)

    @torch.no_grad()
    def save_confidence_vis(self, save_path, mask_tensor):
        """
        保存 Batch 中每一张置信图
        mask_tensor: [B, 1, H, W]
        """
        B = mask_tensor.shape[0]
        base_path, ext = os.path.splitext(save_path)

        for i in range(1):
            m = mask_tensor[i, 0].cpu().numpy()
            
            # Mask 是 0~1，直接乘
            m_uint8 = (m * 255).clip(0, 255).astype(np.uint8)
            m_color = cv2.applyColorMap(m_uint8, cv2.COLORMAP_JET)
            
            # 在图上写上该样本的均值
            mean_conf = m.mean()
            # cv2.putText(m_color, f"Mean: {mean_conf:.4f}", (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            VIS_SIZE = (512, 256)
            m_color = cv2.resize(m_color, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
            
            # 保存
            cv2.imwrite(f"{base_path}_b{i}{ext}", m_color)
 
    def load_ref_data_from_indices(self, indices):
            """
            indices: [B, K]
            从元数据中加载图片及其预计算好的深度图
            """
            B, K = indices.shape
            batch_imgs = []
            batch_depths = []
            
            for b in range(B):
                sample_imgs = []
                sample_depths = []
                for k in range(K):
                    path = self.metadata[indices[b, k]]
                    # 加载并预处理 (这里要和你 Dataset 的 transform 保持一致)
                    img, depth = self.load_single_ref(path) 
                    sample_imgs.append(img)
                    sample_depths.append(depth)
                batch_imgs.append(torch.stack(sample_imgs))
                batch_depths.append(torch.stack(sample_depths))
                
            return torch.stack(batch_imgs), torch.stack(batch_depths) 
            
    def load_ref_image_and_depth(self, path):
            # 加载 RGB
            img = Image.open(path).convert("RGB")
            # 应用相同的 Resize 和 Normalize (-1 到 1)
            img_t = self.val_transform(img) # 需要定义好 transform
            
            # 加载深度图
            # 假设深度图存放在原图路径对应的 _depth 文件夹，且格式为 .npy
            depth_path = path.replace("image_2", "depth").replace(".png", ".npy")
            if os.path.exists(depth_path):
                depth_np = np.load(depth_path)
                depth_t = torch.from_numpy(depth_np).unsqueeze(0).float()
                # 归一化到 0-1
                depth_t = (depth_t - depth_t.min()) / (depth_t.max() - depth_t.min() + 1e-6)
            else:
                # 如果没找到深度图，则用全 0 占位
                depth_t = torch.zeros((1, 256, 512))
    def _get_precomputed_depth_path(self, img_path):
            """ 将图像路径映射到对应的预计算 .npy 深度文件路径 """
            abs_p = os.path.abspath(img_path)
            # 匹配之前的映射逻辑
            if "KITTI" in abs_p:
                d_p = abs_p.replace("KITTI", "KITTI_depth")
            elif "cityscape" in abs_p:
                d_p = abs_p.replace("cityscape_dataset", "cityscape_dataset_depth")
            elif "InStereo2K" in abs_p or "instereo2k" in abs_p:
                d_p = abs_p.replace("InStereo2K", "InStereo2K_depth").replace("instereo2k", "instereo2k_depth")
            else:
                d_p = abs_p
                
            # 统一后缀为 .npy
            return os.path.splitext(d_p)[0] + ".npy"     

    @torch.no_grad()
    def fetch_retrieved_data(self, indices):
        """
        根据 FAISS 返回的索引，从硬盘加载对应的参考图和深度图。
        严格匹配 PairKitti 的几何预处理逻辑：
        1. KITTI: CenterCrop(370, 740) -> Resize(256, 512)
        2. 其他: Resize(256, 512)
        3. 归一化: float/255 * 2 - 1
        """
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        
        B, K = indices.shape
        all_imgs = []
        all_depths = []
        target_size = (256, 512) # 必须与主图尺寸一致

        for b in range(B):
            batch_imgs = []
            batch_depths = []
            for k in range(K):
                # 1. 索引合法性检查
                idx = indices[b, k]
                if idx < 0 or idx >= len(self.metadata):
                    # 检索失败时的兜底（填充全黑图和零深度）
                    batch_imgs.append(torch.full((3, 256, 512), -1.0))
                    batch_depths.append(torch.zeros((1, 256, 512)))
                    continue
                
                img_path = self.metadata[idx]
                abs_img_path = os.path.abspath(img_path)
                
                # 2. 加载原始 RGB 图像
                img = Image.open(img_path).convert("RGB")
                
                # 3. 动态匹配预计算深度图路径 (.npy)
                if "KITTI" in abs_img_path:
                    depth_path = abs_img_path.replace("KITTI", "KITTI_depth")
                elif "cityscape" in abs_img_path:
                    depth_path = abs_img_path.replace("cityscape_dataset", "cityscape_dataset_depth")
                elif "InStereo2K" in abs_img_path or "instereo2k" in abs_img_path:
                    depth_path = abs_img_path.replace("InStereo2K", "InStereo2K_depth").replace("instereo2k", "instereo2k_depth")
                else:
                    depth_path = abs_img_path
                
                depth_path = os.path.splitext(depth_path)[0] + ".npy"

                # 4. 加载原始深度数据
                if os.path.exists(depth_path):
                    d_np = np.load(depth_path)
                    depth = torch.from_numpy(d_np).unsqueeze(0).float() # [1, H, W]
                else:
                    # 找不到深度时的占位（使用 KITTI 典型尺寸）
                    depth = torch.zeros((1, 375, 1242))

                # 5. 【核心】条件几何变换（必须与建库时的特征提取完全对齐）
                if "KITTI" in img_path.upper():
                    # KITTI 序列: 先中心裁剪 (370x740) 以保证长宽比一致
                    img = TF.center_crop(img, (370, 740))
                    depth = TF.center_crop(depth, (370, 740))
                
                # 6. 统一缩放到目标分辨率
                img = TF.resize(img, target_size, interpolation=InterpolationMode.BICUBIC)
                depth = TF.resize(depth, target_size, interpolation=InterpolationMode.BILINEAR)

                # 7. 数据归一化 (匹配你的数据集处理逻辑)
                # RGB: [0, 255] -> [-1, 1]
                img_t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
                img_t = img_t * 2 - 1
                
                # Depth: 线性归一化到 [0, 1]
                d_min, d_max = depth.min(), depth.max()
                if d_max - d_min > 1e-6:
                    depth_t = (depth - d_min) / (d_max - d_min)
                else:
                    depth_t = depth
                
                batch_imgs.append(img_t)
                batch_depths.append(depth_t)
            
            all_imgs.append(torch.stack(batch_imgs))
            all_depths.append(torch.stack(batch_depths))

        # 8. 拼成 Batch Tensor 并搬运到显卡
        res_imgs = torch.stack(all_imgs).to(self.device)
        res_depths = torch.stack(all_depths).to(self.device)
        
        return res_imgs, res_depths   

    # def set_rate_interpolation(self, ckpt_path_1, ckpt_path_2, rho):
    #         """
    #         根据比例 rho 在两个已训练好的码率点之间插值
    #         rho = 0: 完全使用 ckpt_1 (低码率)
    #         rho = 1: 完全使用 ckpt_2 (高码率)
    #         """
    #         state_1 = torch.load(ckpt_path_1)['state_dict']
    #         state_2 = torch.load(ckpt_path_2)['state_dict']
            
    #         # 提取两个量化器的参数
    #         w1, b1 = state_1['preprocess_model.stanh_y.w'], state_1['preprocess_model.stanh_y.b']
    #         w2, b2 = state_2['preprocess_model.stanh_y.w'], state_2['preprocess_model.stanh_y.b']
            
    #         # 执行线性插值
    #         w_interp = (1 - rho) * w1 + rho * w2
    #         b_interp = (1 - rho) * b1 + rho * b2
            
    #         # 将插值后的参数加载到当前模型
    #         self.preprocess_model.stanh_y.w.data.copy_(w_interp)
    #         self.preprocess_model.stanh_y.b.data.copy_(b_interp)
            
    #         print(f">>> Rate adjusted to rho={rho:.2f} via interpolation.")

    @torch.no_grad()
    def save_multi_retrieval_vis(self, save_path, target_img, anchor_emb, vis_storage, attn_weights, sim_scores):
        import cv2
        import numpy as np
        K = len(vis_storage)
        H, W = 256, 512
        
        # 定义每个参考图的代表色 (用于第一列的决策地图)
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]

        def draw_label(img, text, pos, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0)):
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(img, (pos[0]-5, pos[1]-th-10), (pos[0]+tw+5, pos[1]+bl+5), bg_color, -1)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

        def to_cv2(tensor, mode='rgb'):
            t = tensor.detach().cpu()
            if mode == 'depth':
                arr = (np.clip(t.squeeze().numpy(), 0, 1) * 255).astype(np.uint8)
                return cv2.applyColorMap(cv2.resize(arr, (W, H)), cv2.COLORMAP_VIRIDIS)
            elif mode == 'sim': # 可视化相似度曲面 (使用 HOT 色图体现“凸点”)
                arr = t.squeeze().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                return cv2.applyColorMap((cv2.resize(arr, (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            elif mode == 'mask': # 可视化硬选择掩码
                arr = (t.squeeze().numpy() * 255).astype(np.uint8)
                return cv2.applyColorMap(cv2.resize(arr, (W, H)), cv2.COLORMAP_BONE)
            else: # RGB
                arr = t.permute(1, 2, 0).numpy()
                arr = (np.clip((arr + 1.0) / 2.0, 0, 1) * 255).astype(np.uint8)
                return cv2.cvtColor(cv2.resize(arr, (W, H)), cv2.COLOR_RGB2BGR)

        # --- 1. 第一列：Target + 决策决策地图 ---
        img_target = to_cv2(target_img)
        cv2.putText(img_target, "TARGET", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4) # 加粗黑体
        cv2.putText(img_target, "TARGET", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1)
        
        # 制作 Pixel Selection Map (主图像素选择了哪张参考图)
        best_idx = torch.argmax(sim_scores, dim=0).numpy()
        sel_map = np.zeros((best_idx.shape[0], best_idx.shape[1], 3), dtype=np.uint8)
        for k in range(K):
            sel_map[best_idx == k] = colors[k % len(colors)]
        img_sel = cv2.resize(sel_map, (W, H), interpolation=cv2.INTER_NEAREST)
        draw_label(img_sel, "Selection Decision Map", (20, 45), bg_color=(50, 50, 50))
        
        # 补齐仪表盘高度 (1+1+3)
        main_col = np.vstack([img_target, img_sel, np.ones((H*3, W, 3), np.uint8)*255])

        # --- 2. 准备 SI 参考列 ---
        si_cols = []
        for i in range(K):
            data = vis_storage[i]
            # Row 1: RGB
            r_rgb = to_cv2(data['ref_img'])
            draw_label(r_rgb, f"Source {i}", (20, 45))
            
            # Row 2: Depth
            r_depth = to_cv2(data['ref_depth'], mode='depth')
            draw_label(r_depth, "Source Depth", (20, 40), font_scale=0.5)
            
            # Row 3: Flow
            f_np = data['flow_field'].numpy()
            mag, ang = cv2.cartToPolar(f_np[...,0], f_np[...,1])
            hsv = np.zeros((32, 64, 3), np.uint8); hsv[...,0],hsv[...,1],hsv[...,2] = ang*90/np.pi, 255, cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            r_flow = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (W, H))
            draw_label(r_flow, f"a:{data['alpha']:.2f} b:{data['beta']:.2f}", (20, 40), bg_color=(80, 0, 80))

            # Row 4: Matching Surface (曲面可视化)
            r_sim = to_cv2(sim_scores[i], mode='sim')
            draw_label(r_sim, "Matching Surface (Peaks)", (20, 40), bg_color=(80, 0, 0))

            # Row 5: Hard Selection Mask (该图最终贡献区域)
            r_mask = to_cv2(attn_weights[i], mode='mask')
            draw_label(r_mask, "Picked Pixels (WTA)", (20, 40), bg_color=(0, 0, 120))
            
            si_cols.append(np.vstack([r_rgb, r_depth, r_flow, r_sim, r_mask]))

        # --- 3. 最终拼接 ---
        canvas = np.hstack([main_col] + si_cols)
        cv2.imwrite(save_path, canvas)
        
    def get_input(self, batch, k, bs=None, vis_path=None, *args, **kwargs):
        target, z, h, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)  # z为编码段的z_latent
        target_y, z_y, h_y, c_y = super().get_input(batch, 'side', bs=bs, *args, **kwargs) 
        # print("target:",target.shape) # (4,3,256,256)
        # print("z:",x.shape) # (4,4,32,32)
        # print("h:",h.shape) # (4,512,32,32)
        N , _, H, W = z.shape        
        # target = batch [self.first_stage_key], x = latent vector, h = hyper feature, c = txt enbedding
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, f_y = self.apply_condition_encoder(h, h_y) 
        # c_latent 为压缩后传过来h_y经过转化后得到的z_latent(形状一致)
        # side_depth = torch.zeros((N, 1, H*8, W*8)).to(self.device)
        side_img = batch['side'].to(self.device)
        # if 'side_depth' in batch and batch['side_depth'] is not None:
        if batch['side_depth'].max() !=0:
            side_depth = batch['side_depth'].to(self.device).float()
            # print(side_depth)
            # print(side_depth.shape)
        else:
            print("test")
            with torch.no_grad():
                side_depth = self._get_depth_maps(side_img)
        # side_depth = self._get_depth_maps(side_img_tensor)

        do_vis = (vis_path is not None)

        warped_f_y, warped_z_y, vis_data, flow_field = self.lgam(
            depth_side_raw=side_depth, 
            feat_main=guide_hint.detach(), 
            feat_side=f_y, 
            z_main=c_latent.detach(), 
            z_side=z_y,return_vis=do_vis
        ) 
        # print("flow_field:", flow_field.shape) # torch.Size([4, 32, 64, 2])
        # print(c_latent.shape) # [4, 4, 32, 32]
        # print(guide_hint.shape) # [4, 256, 32, 32]
        # guide_hint = g_s(h) , c_latent = conv(g_s(h))
        # guide_hint = guide_hint + warped_f_y
        # c_latent = warped_z_y+c_latent
        c_latent_pre=c_latent
        guide_hint_pre=guide_hint

        feat_cat = torch.cat([guide_hint, warped_f_y], dim=1)
        guide_hint = guide_hint + self.fusion_feature(feat_cat)
        
        # 2. (Residual Cat+Conv)
        latent_cat = torch.cat([c_latent, warped_z_y], dim=1)
        confidence_mask = self.fusion_gate(latent_cat)
        # c_latent = c_latent + self.fusion_latent(latent_cat)
        c_latent = confidence_mask*c_latent + self.fusion_latent(latent_cat)
# -----------------------original-----------------------------------------------------------------------
        if do_vis:
        # if vis_path is not None:
            self.save_debug_vis(
                save_path=vis_path,
                z_main=c_latent_pre,   # 原始主图 Latent
                z_warp=warped_z_y,     # Warp后边图 Latent
                z_fused=c_latent,      # 融合后 Latent
                f_main=guide_hint_pre, # 原始主图 Feature
                f_warp=warped_f_y,     # Warp后边图 Feature
                f_fused=guide_hint     # 融合后 Feature
            )
            lgam_path = vis_path.replace(".png", "_lgam.png")
            self.save_lgam_debug(lgam_path, vis_data)
           
            conf_vis_path = vis_path.replace(".png", "_conf.png")
            self.save_confidence_vis(conf_vis_path, confidence_mask)

        num_pixels = N * H * W * 64
        bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
        q_bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in q_likelihoods)
            
        return z, dict(c_crossattn=[c], c_latent=[c_latent], bpp=bpp, q_bpp=q_bpp, emb_loss=emb_loss, guide_hint=guide_hint, target=target,lgam_params=vis_data,flow_field=flow_field, warped_z_y=warped_z_y, z_y=z_y,side_image=side_img)
               
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        guide_hint = cond['guide_hint']

        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt, guide_hint=guide_hint, base_model=diffusion_model)
        
        return eps
    
    def apply_model_unconditional(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        eps = self.control_model.forward_unconditional(
            x=x_noisy, timesteps=t, context=cond_txt, base_model=diffusion_model)
        
        return eps
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)
 
 
    def frequency_fusion(self, img_high_quality, img_high_fidelity, gamma=0.5):
            # 1. FFT
            # fft_content = torch.fft.rfft2(img_high_fidelity, norm='ortho')
            # fft_sample = torch.fft.rfft2(img_high_quality, norm='ortho')
            
            # 2. 复数域插值 (Complex Interpolation)
            # 这在数学上等价于空间域的线性插值：(1-g)*A + g*B
            # 但我们可以在这里引入频域的 Trick，比如低频部分倾向于 content
            
            # 方案 A: 简单复数插值 (端点严格一致)
            # gamma=0 -> fft_content; gamma=1 -> fft_sample
            target_fft = (1 - gamma) * img_high_fidelity + gamma * img_high_quality
            
            # 方案 B (推荐): 幅度插值 + 相位混合 (Phase Mixing)
            # 如果你依然想利用 Fidelity 的结构稳定性，但希望 gamma=1 时能还原 Perception
            """
            amp_content = torch.abs(fft_content)
            amp_sample = torch.abs(fft_sample)
            phase_content = torch.angle(fft_content)
            phase_sample = torch.angle(fft_sample)
            
            # 幅度插值
            target_amp = (1 - gamma) * amp_content + gamma * amp_sample
            
            # 相位处理：这是难点。简单的线性插值会导致相位抵消。
            # 我们可以只在 gamma > 0.8 (接近纯感知) 时才引入 sample 的相位
            # 或者：直接使用简单的复数插值（方案 A），它自然包含了幅度和相位的变化
            """

            # 3. 逆变换
            # img_fused = torch.fft.irfft2(target_fft, s=img_high_fidelity.shape[-2:], norm='ortho')
            
            return target_fft

    
    @torch.no_grad()
    def log_images(self, batch, sample_steps=5, bs=2,fidelity_ratio=1.0): # fidelity_ratio越大感知效果越好
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"] + 0.003418 # 14 / 64**2
        bpp_img = [f'{bpp:2f}']*4
        c_latent = c["c_latent"][0]
        guide_hint = c['guide_hint']
        target = c["target"]
        c1 = c["c_crossattn"][0]

        log["target"] = (target + 1) / 2
        log["vae_rec"] = (self.decode_first_stage(z) + 1) / 2
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        # log["side_image"]= (c["side_image"]+1) / 2     
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_crossattn": [c1], "c_latent": [c_latent], "guide_hint":guide_hint},
            steps=sample_steps if not self.is_refine else self.fixed_step,
        )

        x_samples = self.decode_first_stage(samples)
        # x_samples = self.decode_first_stage(z_final)
        # if self.train_distortion_decoder_only:
        #     # log["samples"]=(self.distortion_decoder(c_latent) + 1) / 2
        #     # log["samples"]=(self.distortion_decoder(z) + 1) / 2
            
        #     # x_fidelity = (self.distortion_decoder(c_latent) + 1) / 2
            
        #     # latent_fidelity = self.distortion_decoder(c_latent, c["z_y"], c["flow_field"])      
        #     # x_fidelity = (self.decode_first_stage_with_grad(latent_fidelity) + 1)/2           
        #     x_fidelity = (self.distortion_decoder(c_latent, c["z_y"], c["flow_field"]) + 1) / 2
        #     x_perception = (x_samples + 1) / 2
        #     if fidelity_ratio <= 0.0:
        #         x_final = x_fidelity
        #     elif fidelity_ratio >= 1.0:
        #         x_final = x_perception
        #     else:
        #         x_final = self.frequency_fusion(x_perception, x_fidelity, fidelity_ratio)  
        #     log["samples"] = x_final 
                  
        # else:
        log["samples"] = (x_samples + 1) / 2

        return log, bpp
    
    @torch.no_grad()
    def sample_log(self, cond, steps):
        x_T = cond["c_latent"][0]
        b, c, h, w = x_T.shape
        shape = (b, self.channels, h, w)
        t = torch.ones((b,)).long().to(self.device) * self.used_timesteps - 1
        noise = None
        noise = default(noise, lambda: torch.randn_like(x_T))
        x_T = self.q_sample(x_start=x_T, t=t, noise=noise)

        if self.is_refine:
            samples = self.sampler.sample(
            steps, shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None, x_T=x_T
            )
        else:
            sampler = SpacedSampler(self)
            samples = sampler.sample(
                steps, shape, cond, unconditional_guidance_scale=1.0,
                unconditional_conditioning=None, x_T=x_T
            )
        return samples
    
    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     params = list(self.control_model.parameters()) + list(self.preprocess_model.parameters())

    #     if not self.sd_locked:
    #         params += list(self.model.diffusion_model.output_blocks.parameters())
    #         params += list(self.model.diffusion_model.out.parameters())
    #     opt = torch.optim.AdamW(params, lr=lr)

    #     return opt
    def configure_optimizers(self):
            lr = self.learning_rate
    
            params = list(self.control_model.parameters()) + list(self.preprocess_model.parameters())

            if not self.sd_locked:
                print("training unet 11111111111111111111111")
                params = params + list(self.model.diffusion_model.input_blocks.parameters())
                params = params + list(self.model.diffusion_model.middle_block.parameters())
                params = params + list(self.model.diffusion_model.output_blocks.parameters())
                params = params + list(self.model.diffusion_model.out.parameters())

            params = params + list(self.fusion_feature.parameters())
            params = params + list(self.fusion_latent.parameters())
            params = params + list(self.lgam.parameters())
            params = params + list(self.fusion_gate.parameters())
            # params += list(self.model.diffusion_model.parameters())
            opt = torch.optim.AdamW(params, lr=lr)
            return opt    
    def forward(self, x, c, *args, **kwargs):
        if self.is_refine:
            t = torch.ones((x.shape[0],)).long().to(self.device) * self.used_timesteps - 1
        else:
            t = torch.randint(0, self.used_timesteps, (x.shape[0],)).long().to(self.device)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:  # TODO: drop this option
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'
        c_latent = cond['c_latent'][0] 

        if not self.is_refine:
            # --- 1. 基础扩散模型逻辑 (计算感知相关 Loss) ---
            noise = default(noise, lambda: torch.randn_like(x_start)) + (c_latent - x_start) / self.lamba
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_output = self.apply_model(x_noisy, t, cond)

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = x_start
                model_output = self._predict_xstart_from_eps(x_noisy, t, model_output)
            elif self.parameterization == "v":
                target = self.get_v(x_start, noise, t)
            else:
                raise NotImplementedError()

            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})

            logvar_t = self.logvar[t].to(self.device)
            loss = self.l_guide_weight * (loss_simple / torch.exp(logvar_t) + logvar_t).mean()

            # 码率与约束 Loss
            loss_bpp = cond['bpp']
            guide_bpp = cond['q_bpp']
            loss_emb = cond['emb_loss']
            loss_guide = self.get_loss(c_latent, x_start) # 潜变量空间失真

            loss_dict.update({
                f'{prefix}/l_bpp': loss_bpp.mean(),
                f'{prefix}/q_bpp': guide_bpp.mean(),
                f'{prefix}/l_emb': loss_emb.mean(),
                f'{prefix}/l_guide': loss_guide.mean()
            })

            # 组装标准训练模式下的总 Loss
            loss = loss + self.l_bpp_weight * (loss_bpp + loss_emb)
            loss = loss + self.l_guide_weight * loss_guide

            # --- 2. 核心修改：固定权重只训练量化器 (STanH 衍生阶段) ---
            if self.train_distortion_decoder_only:
                """
                在此模式下，我们忽略扩散模型的梯度，
                直接优化：Loss = Rate + lambda * Distortion
                """
                # A. 计算像素级失真 (Distortion)
                # 使用带梯度的 decode 将 STanH 量化后的 latent 转回图像
                # 这样梯度才能传回给 STanH 的参数 w 和 b
                recon_img = self.decode_first_stage(c_latent)
                target_img = cond['target'] # 原图
                
                # 计算 MSE
                loss_mse_pixel = self.get_loss(recon_img, target_img, mean=True)
                
                # B. 计算率 (Rate)
                # 直接使用算出的 bpp
                current_bpp = loss_bpp.mean()

                # C. 组装 RD Loss
                # 这里的 l_guide_weight 充当了传统压缩中的 lambda
                # 公式：L = BPP + lambda * MSE
                loss = current_bpp + self.l_guide_weight * loss_mse_pixel

                # 更新日志字典
                loss_dict.update({
                    f'{prefix}/l_rd_mse': loss_mse_pixel,
                    f'{prefix}/loss': loss
                })
        else:
            b, c, h, w = c_latent.shape
            shape = (b, self.channels, h, w)
            noise = default(noise, lambda: torch.randn_like(c_latent))
            x_T = self.q_sample(x_start=c_latent, t=t, noise=noise) # 按照时间步t加噪获得加噪后的潜变量
            # noise = default(noise, lambda: torch.randn_like(warped_latent_y))
            # x_T = self.q_sample(x_start=warped_latent_y, t=t, noise=noise) # 按照时间步t加噪获得加噪后的潜变量
            steps = self.fixed_step

            samples = self.sampler.sample_grad(
                steps, shape, cond, unconditional_guidance_scale=1.0,
                unconditional_conditioning=None, x_T=x_T
                ) # 扩散模型去噪过程,获得去噪后的潜变量
            
            model_output = self.decode_first_stage_with_grad(samples) #去噪后的潜变量经过VAE-Decoder解码获得重构的图像
            target = cond['target']

            loss_simple = self.get_loss(samples, c_latent, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
            loss = self.l_guide_weight * loss_simple.mean() #去噪损失

            loss_mse = self.get_loss(model_output, target, mean=False).mean([1, 2, 3]) 
            loss_dict.update({f'{prefix}/l_mse': loss_mse.mean()})
            loss = self.l_guide_weight * loss_mse.mean() #重建失真损失

            loss_lpips = self.perceptual_loss(model_output, target)
            loss_dict.update({f'{prefix}/l_lpips': loss_lpips.mean()})
            loss = loss + self.l_guide_weight * loss_lpips * 0.5 # 重建感知损失

            # compression
            loss_guide = self.get_loss(c_latent, x_start) # 潜变量变换对齐损失,变相等价于压缩后的特征损失
            loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
            loss = loss + self.l_guide_weight * loss_guide 
            loss_dict.update({f'{prefix}/loss': loss})

            loss_bpp = cond['bpp']
            guide_bpp = cond['q_bpp']
            loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
            loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
            loss = loss + self.l_bpp_weight * loss_bpp

            loss_emb = cond['emb_loss'] # codebook loss
            loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()})
            loss = loss + self.l_bpp_weight * loss_emb

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = []
        # cancel annotation while memory is not enough
        # if self.is_refine:
        #     return out
        log, bpp = self.log_images(batch,bs=None)
        out.append(bpp.cpu())
        # save images
        save_dir = os.path.join(self.logger.save_dir, "validation", f'{self.global_step}')
        os.makedirs(save_dir, exist_ok=True)
        image = log["samples"].detach().cpu()
        image = image.numpy().squeeze().transpose(1,2,0)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f'{batch_idx}.png')
        Image.fromarray(image).save(path)

        target = log["target"].detach().cpu()
        target = target.numpy().squeeze().transpose(1,2,0)
        target = (target * 255).clip(0, 255).astype(np.uint8)

        metric_data = [img2tensor(image).unsqueeze(0) / 255.0, img2tensor(target).unsqueeze(0) / 255.0]

        for name, _ in self.calculate_metrics.items():
            out.append(self.metric_funcs[name](*metric_data))

        
        return out
    
    def on_validation_epoch_start(self):
        self.preprocess_model.quantize.reset_usage()
        return super().on_validation_epoch_start()
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        # cancel annotation while memory is not enough
        # if self.is_refine:
        #     return None
        outputs = np.array(outputs)
        avg_out = sum(outputs)/len(outputs)
        self.log("avg_bpp", avg_out[0],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        usage = self.preprocess_model.quantize.get_usage()
        self.log("usage", usage,
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        for i, (name, _) in enumerate(self.calculate_metrics.items()):
            self.log(f"avg_{name}", avg_out[i+1],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
    def load_preprocess_ckpt(self, ckpt_path_pre):
        ckpt = torch.load(ckpt_path_pre)
        self.preprocess_model.load_state_dict(ckpt)
        print(['CONTROL WEIGHTS LOADED'])
        
    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        ckpt_base = torch.load(path)  # load the base model checkpoints

        if synch_control:
            # add copy for control_model weights from the base model
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim]
                            else:
                                dim = 0
                                control_dim_0 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                dim = 1
                                control_dim_1 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim_0, :control_dim_1, ...]
                        else:
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]
            
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')