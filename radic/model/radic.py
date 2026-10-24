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
from torch.utils.checkpoint import checkpoint as pytorch_checkpoint
from torch.utils.checkpoint import checkpoint
from ldm.modules.diffusionmodules.util import conv_nd, linear, zero_module, timestep_embedding, checkpoint
from .spaced_sampler_relay import SpacedSampler
from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock as ResBlock_orig, Downsample, Upsample, AttentionBlock, TimestepBlock
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from .lpips import LPIPS
from .si_supervision import reference_confidence_target, masked_reference_attention, hyper_index_bpp
from gallery import load_gallery

class LGAM(nn.Module):

    def __init__(self, feature_dim=256, latent_dim=4):
        super().__init__()
        self.global_param_net = nn.Sequential(nn.Conv2d(feature_dim * 2, 128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.LeakyReLU(0.1), nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
        self.res_flow_net = nn.Sequential(nn.Conv2d(feature_dim * 2 + 2, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1), nn.Conv2d(64, 3, kernel_size=3, padding=1))

    def get_grid(self, B, H, W, device):
        xx = ((torch.arange(W, device=device) + 0.5) * (2.0 / W) - 1.0).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = ((torch.arange(H, device=device) + 0.5) * (2.0 / H) - 1.0).view(1, 1, H, 1).expand(B, -1, -1, W)
        return torch.cat([xx, yy], dim=1)

    def forward(self, depth_side_raw, feat_main, feat_side, z_main, z_side, return_vis=False):
        (B, C, H, W) = feat_main.shape
        device = feat_main.device
        depth_small = F.interpolate(depth_side_raw, size=(H, W), mode='bilinear', align_corners=False)
        inv_depth = 1.0 / (depth_small + 1e-06)
        minimum = inv_depth.amin(dim=(2, 3), keepdim=True)
        maximum = inv_depth.amax(dim=(2, 3), keepdim=True)
        inv_depth_norm = (inv_depth - minimum) / (maximum - minimum + 1e-06)
        params = self.global_param_net(torch.cat([feat_main, feat_side], dim=1))
        alpha = F.softplus(params[:, 0]).view(B, 1, 1, 1) + 0.05
        beta = torch.tanh(params[:, 1]).view(B, 1, 1, 1)
        flow_coarse_x = alpha * inv_depth_norm + beta
        refine_input = torch.cat([feat_main, feat_side, flow_coarse_x, inv_depth_norm], dim=1)
        res_output = self.res_flow_net(refine_input)
        delta_flow = res_output[:, :2, ...]
        conf_mask = torch.sigmoid(res_output[:, 2:3, ...])
        final_flow_x = flow_coarse_x + delta_flow[:, 0:1, ...]
        final_flow_y = delta_flow[:, 1:2, ...]
        flow_field = torch.cat([final_flow_x, final_flow_y], dim=1).permute(0, 2, 3, 1)
        base_grid = self.get_grid(B, H, W, device).permute(0, 2, 3, 1)
        sample_grid = base_grid + flow_field
        warped_feat_side = F.grid_sample(feat_side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_feat_side = warped_feat_side * conf_mask
        warped_z_side = F.grid_sample(z_side, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_z_side = warped_z_side * conf_mask
        vis_data = {'inv_depth': inv_depth_norm.detach(), 'flow_x': final_flow_x.detach(), 'conf_mask': conf_mask.detach(), 'alpha': alpha.detach().mean(), 'beta': beta.detach().mean()}
        return (warped_feat_side, warped_z_side, vis_data, flow_field)

class MultiReferenceFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Conv2d(channels, channels // 4, 1)
        self.k_proj = nn.Conv2d(channels, channels // 4, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        num_groups = min(channels, 32)
        if channels % num_groups != 0:
            num_groups = 1
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(num_groups, channels), nn.SiLU())

    def forward(self, f_main, f_refs_stack, valid=None):
        (B, K, C, H, W) = f_refs_stack.shape
        Cp = C // 4
        q = self.q_proj(f_main)
        k = self.k_proj(f_refs_stack.view(-1, C, H, W)).view(B, K, Cp, H, W)
        v = self.v_proj(f_refs_stack.view(-1, C, H, W)).view(B, K, C, H, W)
        sim = torch.sum(q.unsqueeze(1) * k, dim=2) / Cp ** 0.5
        if valid is None:
            valid = torch.ones(B, K, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(~valid[:, :, None, None], torch.finfo(sim.dtype).min)
        best_idx = torch.argmax(sim, dim=1)
        mask_hard = F.one_hot(best_idx, num_classes=K).permute(0, 3, 1, 2).float()
        mask_hard = mask_hard * valid[:, :, None, None]
        attn_soft = masked_reference_attention(sim, valid)
        mask_ste = mask_hard + attn_soft - attn_soft.detach()
        f_side = torch.sum(v * mask_ste.unsqueeze(2), dim=1)
        fused = self.refine(f_side) * valid.any(1)[:, None, None, None]
        return (fused, mask_hard, sim)

class NoiseEstimator(nn.Module):

    def __init__(self, image_size, in_channels, model_channels, out_channels, hint_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=False, use_linear_in_transformer=False, control_model_ratio=1.0, learn_embedding=True, control_scale=1.0):
        super().__init__()
        self.learn_embedding = learn_embedding
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.control_scale = control_scale
        base_model = UNetModel(image_size=image_size, in_channels=in_channels, model_channels=model_channels, out_channels=out_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth, context_dim=context_dim, n_embed=n_embed, legacy=legacy, use_linear_in_transformer=use_linear_in_transformer)
        self.control_model = ControlModule(image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels, out_channels=out_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult, conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order, use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth, context_dim=context_dim, n_embed=n_embed, legacy=legacy, use_linear_in_transformer=use_linear_in_transformer, control_model_ratio=control_model_ratio)
        self.enc_zero_convs_out = nn.ModuleList([])
        self.middle_block_out = None
        self.middle_block_in = None
        self.dec_zero_convs_out = nn.ModuleList([])
        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}
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
        self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])
        self.dec_zero_convs_out.append(self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1]))
        for i in range(1, len(ch_inout_ctr['enc'])):
            self.dec_zero_convs_out.append(self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1]))
        for i in range(len(ch_inout_ctr['enc'])):
            self.enc_zero_convs_out.append(self.make_zero_conv(in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1]))
        scale_list = [1.0] * len(self.enc_zero_convs_out) + [1.0] + [1.0] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list) * self.control_scale)

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0)))

    def forward(self, x, guide_hint, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.control_model.time_embed(t_emb)
        emb_base = base_model.time_embed(t_emb)
        h_base = x.type(base_model.dtype)
        h_ctr = torch.cat((h_base, guide_hint), dim=1)
        hs_base = []
        hs_ctr = []
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list * self.control_scale)
        for (module_base, module_ctr) in zip(base_model.input_blocks, self.control_model.input_blocks):
            h_base = module_base(h_base, emb_base, context)
            h_ctr = module_ctr(h_ctr, emb, context)
            h_base = h_base + next(it_enc_convs_out)(h_ctr, emb) * next(scales)
            hs_base.append(h_base)
            hs_ctr.append(h_ctr)
        h_base = base_model.middle_block(h_base, emb_base, context)
        h_ctr = self.control_model.middle_block(h_ctr, emb, context)
        h_base = h_base + self.middle_block_out(h_ctr, emb) * next(scales)
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
        for module_base in base_model.input_blocks:
            h_base = module_base(h_base, emb_base, context)
            hs_base.append(h_base)
        h_base = base_model.middle_block(h_base, emb_base, context)
        for module_base in base_model.output_blocks:
            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)
        return base_model.out(h_base)

class ControlModule(nn.Module):

    def __init__(self, image_size, in_channels, model_channels, hint_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, control_model_ratio=1.0):
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
                raise ValueError('provide num_res_blocks either as an int (globally constant) or as a list/tuple (per-level) with the same length as channel_mult')
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f'Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. This option has LESS priority than attention_resolutions {attention_resolutions}, i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, attention will still not be set.')
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
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        model_channels = int(model_channels * control_model_ratio)
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == 'continuous':
                print('setting up linear c_adm embedding layer')
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for (level, mult) in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
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
                        layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order) if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
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
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order) if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch

def find_denominator(number, start):
    if start >= number:
        return number
    while start != 0:
        residual = number % start
        if residual == 0:
            return start
        start -= 1

def normalization(channels):
    return GroupNorm_leq32(find_denominator(channels, 32), channels)

class GroupNorm_leq32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class ResBlock(TimestepBlock):

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            (in_rest, in_conv) = (self.in_layers[:-1], self.in_layers[-1])
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
            (out_norm, out_rest) = (self.out_layers[0], self.out_layers[1:])
            (scale, shift) = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
import numpy as np

class RADIC(LatentDiffusion):

    def __init__(self, control_stage_config: Mapping[str, Any], sd_locked: bool, is_refine: bool, fixed_step: int, learning_rate: float, l_bpp_weight: float, l_guide_weight: float, used_timesteps: int, sync_path: str, synch_control: bool, ckpt_path_pre: str, preprocess_config: Mapping[str, Any], calculate_metrics: Mapping[str, Any], train_distortion_decoder_only: bool, n_refs=1, train_n_refs=3, retrieval_candidates=20, gallery_root='retrieval_db', *args, **kwargs) -> 'RADIC':
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.preprocess_model = instantiate_from_config(preprocess_config)
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
        for (_, opt) in calculate_metrics.items():
            mopt = opt.copy()
            name = mopt.pop('type', None)
            mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
        self.lamba = self.sqrt_recipm1_alphas_cumprod[self.used_timesteps - 1]
        if self.is_refine:
            self.sampler = SpacedSampler(self)
            self.perceptual_loss = LPIPS(pnet_type='vgg')
        self.lgam = LGAM()
        print('Loading Depth Anything V3...')
        self.depth_model = DepthAnything3.from_pretrained('depth_anything_3/DA3NESTED-GIANT-LARGE')
        self.depth_model.eval()
        for param in self.depth_model.parameters():
            param.requires_grad = False
        self.fusion_feature = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1), nn.GroupNorm(32, 256), nn.SiLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.fusion_latent = nn.Sequential(nn.Conv2d(4 * 2, 16, kernel_size=1), nn.SiLU(), nn.Conv2d(16, 4, kernel_size=1))
        self.fusion_gate = nn.Sequential(nn.Conv2d(4 * 2, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid())
        self.train_distortion_decoder_only = train_distortion_decoder_only
        self.use_retrieval = True
        self.n_refs = int(n_refs)
        self.train_n_refs = int(train_n_refs)
        self.retrieval_candidates = int(retrieval_candidates)
        if min(self.n_refs, self.train_n_refs) < 0 or self.retrieval_candidates < self.train_n_refs:
            raise ValueError('Invalid reference counts')
        self.gallery_root = gallery_root
        self.gallery_split = 'test'
        self.loaded_gallery_split = None
        self.fusion_feature_multi = MultiReferenceFusion(256)
        self.fusion_latent_multi = MultiReferenceFusion(4)
        self.gallery_momentum = 0.99

    def set_gallery(self, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Invalid gallery split')
        if self.loaded_gallery_split != split:
            (self.index, self.metadata) = load_gallery(self.gallery_root, split)
            self.loaded_gallery_split = split
        self.gallery_split = split

    def apply_condition_encoder(self, x, y):
        (c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y, _) = self.preprocess_model(x, y)
        return (c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y)

    @torch.no_grad()
    def apply_condition_compress(self, x, stream_path, H, W):
        (_, h) = self.encode_first_stage(x * 2 - 1)
        h = h * self.scale_factor
        out = self.preprocess_model.compress(h)
        shape = out['shape']
        with Path(stream_path).open('wb') as f:
            write_body(f, shape, out['strings'])
        size = filesize(stream_path)
        bpp = float(size) * 8 / (H * W)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(self, stream_path):
        with Path(stream_path).open('rb') as f:
            (strings, shape) = read_body(f)
        (c_latent, guide_hint) = self.preprocess_model.decompress(strings, shape)
        return (c_latent, guide_hint)

    def _get_depth_maps(self, images_tensor):
        if next(self.depth_model.parameters()).device != images_tensor.device:
            self.depth_model.to(images_tensor.device)
        (B, C, H, W) = images_tensor.shape
        depth_maps = []
        images_np = ((images_tensor + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        with torch.no_grad():
            for i in range(B):
                img_np = images_np[i]
                prediction = self.depth_model.inference([img_np])
                raw_depth = prediction.depth[0]
                if raw_depth.shape != (H, W):
                    raw_depth = cv2.resize(raw_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                depth_maps.append(raw_depth)
        depth_batch = np.stack(depth_maps, axis=0)
        depth_tensor = torch.from_numpy(depth_batch).unsqueeze(1).to(images_tensor.device).float()
        d_min = depth_tensor.flatten(2).min(2)[0].view(B, 1, 1, 1)
        d_max = depth_tensor.flatten(2).max(2)[0].view(B, 1, 1, 1)
        depth_tensor = (depth_tensor - d_min) / (d_max - d_min + 1e-06)
        return depth_tensor

    @torch.no_grad()
    def save_depth_vis(self, save_path, depth_tensor):
        B = depth_tensor.shape[0]
        (base_path, ext) = os.path.splitext(save_path)
        for i in range(B):
            d = depth_tensor[i, 0].cpu().numpy()
            (d_min, d_max) = (d.min(), d.max())
            if d_max - d_min > 1e-06:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
            d_uint8 = (d_norm * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_VIRIDIS)
            VIS_SIZE = (512, 256)
            d_color = cv2.resize(d_color, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f'{base_path}_b{i}{ext}', d_color)

    @torch.no_grad()
    def save_confidence_vis(self, save_path, mask_tensor):
        B = mask_tensor.shape[0]
        (base_path, ext) = os.path.splitext(save_path)
        for i in range(1):
            m = mask_tensor[i, 0].cpu().numpy()
            m_uint8 = (m * 255).clip(0, 255).astype(np.uint8)
            m_color = cv2.applyColorMap(m_uint8, cv2.COLORMAP_JET)
            mean_conf = m.mean()
            VIS_SIZE = (512, 256)
            m_color = cv2.resize(m_color, VIS_SIZE, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f'{base_path}_b{i}{ext}', m_color)

    def load_ref_data_from_indices(self, indices):
        (B, K) = indices.shape
        batch_imgs = []
        batch_depths = []
        for b in range(B):
            sample_imgs = []
            sample_depths = []
            for k in range(K):
                path = self.metadata[indices[b, k]]
                (img, depth) = self.load_single_ref(path)
                sample_imgs.append(img)
                sample_depths.append(depth)
            batch_imgs.append(torch.stack(sample_imgs))
            batch_depths.append(torch.stack(sample_depths))
        return (torch.stack(batch_imgs), torch.stack(batch_depths))

    def load_ref_image_and_depth(self, path):
        img = Image.open(path).convert('RGB')
        img_t = self.val_transform(img)
        depth_path = path.replace('image_2', 'depth').replace('.png', '.npy')
        if os.path.exists(depth_path):
            depth_np = np.load(depth_path)
            depth_t = torch.from_numpy(depth_np).unsqueeze(0).float()
            depth_t = (depth_t - depth_t.min()) / (depth_t.max() - depth_t.min() + 1e-06)
        else:
            depth_t = torch.zeros((1, 256, 512))

    def _get_precomputed_depth_path(self, img_path):
        abs_p = os.path.abspath(img_path)
        if 'KITTI' in abs_p:
            d_p = abs_p.replace('KITTI', 'KITTI_depth')
        elif 'cityscape' in abs_p:
            d_p = abs_p.replace('cityscape_dataset', 'cityscape_dataset_depth')
        elif 'InStereo2K' in abs_p or 'instereo2k' in abs_p:
            d_p = abs_p.replace('InStereo2K', 'InStereo2K_depth').replace('instereo2k', 'instereo2k_depth')
        else:
            d_p = abs_p
        return os.path.splitext(d_p)[0] + '.npy'

    @torch.no_grad()
    def fetch_retrieved_data(self, indices):
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        import os
        (B, K) = indices.shape
        all_imgs = []
        all_depths = []
        target_size = (256, 512)
        for b in range(B):
            batch_imgs = []
            batch_depths = []
            for k in range(K):
                idx = indices[b, k]
                if idx < -1 or idx >= len(self.metadata):
                    raise IndexError(idx)
                if idx == -1:
                    batch_imgs.append(torch.full((3, 256, 512), -1.0))
                    batch_depths.append(torch.zeros((1, 256, 512)))
                    continue
                img_path = self.metadata[idx]
                abs_img_path = os.path.abspath(img_path)
                img = Image.open(img_path).convert('RGB')
                if 'KITTI' in abs_img_path:
                    depth_path = abs_img_path.replace('KITTI', 'KITTI_depth')
                elif 'cityscape' in abs_img_path:
                    depth_path = abs_img_path.replace('cityscape_dataset', 'cityscape_dataset_depth')
                elif 'InStereo2K' in abs_img_path:
                    depth_path = abs_img_path.replace('InStereo2K', 'InStereo2K_depth')
                elif 'instereo2k' in abs_img_path:
                    parent_dir = os.path.dirname(abs_img_path)
                    file_name = os.path.basename(abs_img_path)
                    depth_path = os.path.join(parent_dir + '_depth', file_name)
                else:
                    depth_path = abs_img_path
                depth_path = os.path.splitext(depth_path)[0] + '.npy'
                if os.path.exists(depth_path):
                    d_np = np.load(depth_path)
                    depth = torch.from_numpy(d_np).unsqueeze(0).float()
                else:
                    raise FileNotFoundError(depth_path)
                img = TF.center_crop(img, (370, 740))
                depth = TF.center_crop(depth, (370, 740))
                img = TF.resize(img, target_size, interpolation=InterpolationMode.BICUBIC)
                depth = TF.resize(depth, target_size, interpolation=InterpolationMode.BILINEAR)
                img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_t = img_t * 2 - 1
                (d_min, d_max) = (depth.min(), depth.max())
                if d_max - d_min > 1e-06:
                    depth_t = (depth - d_min) / (d_max - d_min)
                else:
                    depth_t = depth
                batch_imgs.append(img_t)
                batch_depths.append(depth_t)
            all_imgs.append(torch.stack(batch_imgs))
            all_depths.append(torch.stack(batch_depths))
        res_imgs = torch.stack(all_imgs).to(self.device)
        res_depths = torch.stack(all_depths).to(self.device)
        return (res_imgs, res_depths)

    @torch.no_grad()
    def save_multi_retrieval_vis(self, save_path, target_img, anchor_emb, vis_storage, attn_weights, sim_scores, pred_mask):
        import cv2
        import numpy as np
        K_total = len(vis_storage)
        if K_total == 0:
            return
        (H, W) = (256, 512)

        def draw_label(img, text, pos, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0)):
            ((tw, th), bl) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(img, (pos[0] - 5, pos[1] - th - 10), (pos[0] + tw + 5, pos[1] + bl + 5), bg_color, -1)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

        def to_cv2(tensor, mode='rgb'):
            t = tensor.detach().cpu()
            if mode == 'depth':
                arr = (np.clip(t.squeeze().numpy(), 0, 1) * 255).astype(np.uint8)
                return cv2.applyColorMap(cv2.resize(arr, (W, H)), cv2.COLORMAP_VIRIDIS)
            elif mode == 'attn':
                arr = t.squeeze().numpy()
                arr = np.clip(arr, 0.0, 1.0)
                return cv2.applyColorMap((cv2.resize(arr, (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            elif mode == 'binary_mask':
                threshold = 1.0 / (K_total + 1)
                arr = (t.squeeze().numpy() > threshold).astype(np.uint8) * 255
                arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                arr = t.permute(1, 2, 0).numpy()
                arr = (np.clip((arr + 1.0) / 2.0, 0, 1) * 255).astype(np.uint8)
                return cv2.cvtColor(cv2.resize(arr, (W, H)), cv2.COLOR_RGB2BGR)

        def flow_to_color(flow_tensor):
            flow_np = flow_tensor.detach().cpu().numpy()
            (mag, ang) = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
            hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return cv2.resize(flow_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        img_target = to_cv2(target_img)
        draw_label(img_target, 'TARGET (Query)', (20, 45), bg_color=(0, 100, 0))
        if pred_mask is not None:
            pm = pred_mask[0] if pred_mask.ndim == 4 else pred_mask
            p_attn_2d = pm.mean(dim=0).detach().cpu().numpy() if pm.ndim == 3 else pm.detach().cpu().numpy()
            p_attn_norm = (p_attn_2d - p_attn_2d.min()) / (p_attn_2d.max() - p_attn_2d.min() + 1e-06)
            attn_color = cv2.applyColorMap((p_attn_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            attn_color = cv2.resize(attn_color, (W, H), interpolation=cv2.INTER_LINEAR)
            target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
            target_bg = cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR)
            img_pred_attn = cv2.addWeighted(target_bg, 0.4, attn_color, 0.65, 0)
        else:
            img_pred_attn = np.zeros((H, W, 3), dtype=np.uint8)
        dash_panel = np.ones((H * 3, W, 3), dtype=np.uint8) * 255
        cv2.putText(dash_panel, 'Reference Utilization Score', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(dash_panel, (20, 60), (W - 20, 60), (200, 200, 200), 2)
        start_y = 100
        step_y = (H * 3 - start_y) // max(K_total, 1)
        for i in range(K_total):
            y_b = start_y + i * step_y
            score = attn_weights[i].mean().item() if i < len(attn_weights) else 0.0
            cv2.putText(dash_panel, f'Ref {i} Score: {score:.3f}', (20, y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            bar_len = int(np.clip(score, 0.0, 1.0) * (W - 150))
            cv2.rectangle(dash_panel, (20, y_b + 10), (20 + bar_len, y_b + 25), (100, 200, 100), -1)
        main_col = np.vstack([img_target, img_pred_attn, dash_panel])
        si_cols = []
        for i in range(K_total):
            data = vis_storage[i]
            img_flow = flow_to_color(data['flow_field'])
            rows = [to_cv2(data['ref_img']), to_cv2(data['ref_depth'], 'depth'), img_flow, to_cv2(attn_weights[i], 'attn'), to_cv2(attn_weights[i], 'binary_mask')]
            draw_label(rows[0], f'Source {i}', (20, 45))
            draw_label(rows[2], f'Flow Field Map', (20, 40), bg_color=(80, 0, 80))
            draw_label(rows[2], f"a:{data.get('alpha', 0):.2f} b:{data.get('beta', 0):.2f}", (20, H - 20), font_scale=0.5, color=(0, 255, 255))
            draw_label(rows[3], f'Final Attention (Abs)', (20, 40), bg_color=(150, 0, 0))
            draw_label(rows[4], f'Picked Pixels (Binary)', (20, 40), bg_color=(100, 100, 100))
            si_cols.append(np.vstack(rows))
        canvas = np.hstack([main_col] + si_cols)
        cv2.imwrite(save_path, canvas)

    def get_input(self, batch, k, bs=None, vis_path=None, *args, **kwargs):
        (target, z, h, c) = super().get_input(batch, self.first_stage_key, *args, bs=bs, **kwargs)
        (c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y, pred_mask) = self.preprocess_model(h, h)
        f_anchor = self.preprocess_model.get_embedding(guide_hint)
        hyper_shape = self.preprocess_model.hyper_index_shape
        B = guide_hint.shape[0]
        do_vis = vis_path is not None
        with torch.no_grad():
            query_feat_fixed = f_anchor.detach()
        side_gt = batch['side'][:B].to(self.device)
        if self.training:
            if self.train_n_refs:
                self.set_gallery('train')
                if self.index.d != query_feat_fixed.shape[1]:
                    raise ValueError('Rebuild the gallery index with the current model descriptor.')
            counts = [random.randint(0, self.train_n_refs) for _ in range(B)]
            K_num = max(counts)
            if K_num:
                candidate_count = min(self.retrieval_candidates, self.index.ntotal)
                if candidate_count < K_num:
                    raise ValueError('The gallery contains fewer entries than the requested K.')
                active = np.flatnonzero(np.asarray(counts) > 0)
                (_, candidates) = self.index.search(query_feat_fixed[active].cpu().numpy().astype('float32'), candidate_count)
                indices = np.full((B, K_num), -1, dtype=np.int64)
                for (row, b) in enumerate(active):
                    available = [int(i) for i in candidates[row] if i >= 0]
                    if len(set(available)) < counts[b]:
                        raise ValueError('Insufficient unique retrieval candidates')
                    indices[b, :counts[b]] = random.sample(available, counts[b])
                (si_imgs, si_depths) = self.fetch_retrieved_data(indices)
            else:
                (_, C, H, W) = side_gt.shape
                si_imgs = side_gt.new_empty(B, 0, C, H, W)
                si_depths = side_gt.new_empty(B, 0, 1, H, W)
            valid_refs = torch.arange(K_num, device=self.device)[None, :] < torch.tensor(counts, device=self.device)[:, None]
        elif 'multi_side' in batch:
            si_imgs = batch['multi_side'][:B].to(self.device)
            (B, K_num, C, H, W) = si_imgs.shape
            if K_num:
                si_depths = self._get_depth_maps(si_imgs.reshape(B * K_num, C, H, W)).reshape(B, K_num, 1, H, W)
            else:
                si_depths = si_imgs.new_empty(B, 0, 1, H, W)
            valid_refs = torch.ones(B, K_num, device=self.device, dtype=torch.bool)
        else:
            K_num = self.n_refs
            valid_refs = torch.ones(B, K_num, device=self.device, dtype=torch.bool)
            if K_num == 0:
                (_, C, H, W) = side_gt.shape
                si_imgs = side_gt.new_empty(B, 0, C, H, W)
                si_depths = side_gt.new_empty(B, 0, 1, H, W)
            else:
                self.set_gallery(self.gallery_split)
                if self.index.d != query_feat_fixed.shape[1] or self.index.ntotal < K_num:
                    raise ValueError('Gallery descriptor or size mismatch')
                (_, indices) = self.index.search(query_feat_fixed.cpu().numpy().astype('float32'), K_num)
                (si_imgs, si_depths) = self.fetch_retrieved_data(indices)
        ref_scores = []
        confidence_maps = []
        (B, total_N) = (si_imgs.shape[0], si_imgs.shape[1])
        (warped_f_list, warped_z_list, si_embs, vis_storage) = ([], [], [], [])
        for i in range(total_N):
            (img_i, depth_i) = (si_imgs[:, i], si_depths[:, i])
            with torch.set_grad_enabled(self.training):
                (_, h_i) = self.encode_first_stage(img_i)
                h_i = self.get_first_stage_encoding(h_i)
                (f_yi, z_yi) = self.preprocess_model.side_encode(h_i)
            with torch.no_grad():
                feat_i_fixed = self.preprocess_model.get_embedding(f_yi).detach()
                global_sim = torch.sum(query_feat_fixed * feat_i_fixed, dim=1, keepdim=True).clamp(0, 1)
                ref_scores.append(global_sim)
            (wf, wz, vis_d, flow) = self.lgam(depth_i, guide_hint, f_yi, c_latent, z_yi, return_vis=do_vis)
            confidence_maps.append(vis_d['conf_mask'].squeeze(1))
            warped_f_list.append(wf)
            warped_z_list.append(wz)
            si_embs.append(self.preprocess_model.get_embedding(f_yi))
            if do_vis:
                vis_storage.append({'ref_img': img_i[0], 'ref_depth': depth_i[0], 'warped_z': wz[0], 'alpha': vis_d['alpha'].item(), 'beta': vis_d['beta'].item(), 'flow_field': flow[0].detach().cpu(), 'embedding': si_embs[-1][0].detach().cpu()})
        if total_N == 0:
            guide_hint = guide_hint
            c_latent = c_latent
            attn_weights = guide_hint.new_zeros(B, 0, *guide_hint.shape[-2:])
            sim_scores = guide_hint.new_zeros(B, 0, *guide_hint.shape[-2:])
            confidence = sim_scores
        else:
            f_stack = torch.stack(warped_f_list, dim=1)
            z_stack = torch.stack(warped_z_list, dim=1)
            (f_fused, attn_weights, sim_scores) = self.fusion_feature_multi(guide_hint, f_stack, valid_refs)
            (z_fused, _, _) = self.fusion_latent_multi(c_latent, z_stack, valid_refs)
            confidence = torch.stack(confidence_maps, dim=1) * valid_refs[:, :, None, None]
            has_ref = valid_refs.any(1)[:, None, None, None]
            cat_f = torch.cat([guide_hint, f_fused], dim=1)
            guide_hint = torch.where(has_ref, guide_hint + self.fusion_feature(cat_f), guide_hint)
            cat_z = torch.cat([c_latent, z_fused], dim=1)
            c_latent = torch.where(has_ref, self.fusion_gate(cat_z) * c_latent + self.fusion_latent(cat_z), c_latent)
        if do_vis:
            self.save_multi_retrieval_vis(vis_path, target[0], f_anchor[0].detach().cpu(), vis_storage, attn_weights[0].detach().cpu(), sim_scores[0].detach().cpu(), pred_mask)
        num_pixels = target.shape[0] * target.shape[-2] * target.shape[-1]
        hyper_bpp = hyper_index_bpp(hyper_shape, self.preprocess_model.codebook_size, num_pixels)
        bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels) for l in likelihoods))
        q_bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels) for l in q_likelihoods))
        bpp = bpp + hyper_bpp
        q_bpp = q_bpp + hyper_bpp
        return (z, dict(hyper_bpp=hyper_bpp, gt_confidence=confidence, c_crossattn=[c], c_latent=[c_latent], bpp=bpp, q_bpp=q_bpp, emb_loss=emb_loss, f_anchor=f_anchor, guide_hint=guide_hint, target=target, gt_attn=masked_reference_attention(sim_scores.detach(), valid_refs), pred_attn=pred_mask))

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        guide_hint = cond['guide_hint']
        eps = self.control_model(x=x_noisy, timesteps=t, context=cond_txt, guide_hint=guide_hint, base_model=diffusion_model)
        return eps

    def apply_model_unconditional(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eps = self.control_model.forward_unconditional(x=x_noisy, timesteps=t, context=cond_txt, base_model=diffusion_model)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([''] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=5, bs=2, fidelity_ratio=1.0, vis_path=None):
        log = dict()
        (z, c) = self.get_input(batch, self.first_stage_key, bs=bs, vis_path=vis_path)
        bpp = c['q_bpp']
        bpp_img = [f'{bpp:2f}'] * 4
        c_latent = c['c_latent'][0]
        guide_hint = c['guide_hint']
        target = c['target']
        c1 = c['c_crossattn'][0]
        log['target'] = (target + 1) / 2
        log['vae_rec'] = (self.decode_first_stage(z) + 1) / 2
        log['text'] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        samples = self.sample_log(cond={'c_crossattn': [c1], 'c_latent': [c_latent], 'guide_hint': guide_hint}, steps=sample_steps if not self.is_refine else self.fixed_step)
        x_samples = self.decode_first_stage(samples)
        log['samples'] = (x_samples + 1) / 2
        return (log, bpp)

    @torch.no_grad()
    def sample_log(self, cond, steps):
        x_T = cond['c_latent'][0]
        (b, c, h, w) = x_T.shape
        shape = (b, self.channels, h, w)
        t = torch.ones((b,)).long().to(self.device) * self.used_timesteps - 1
        noise = None
        noise = default(noise, lambda : torch.randn_like(x_T))
        x_T = self.q_sample(x_start=x_T, t=t, noise=noise)
        if self.is_refine:
            samples = self.sampler.sample(steps, shape, cond, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=x_T)
        else:
            sampler = SpacedSampler(self)
            samples = sampler.sample(steps, shape, cond, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=x_T)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters()) + list(self.preprocess_model.parameters())
        if not self.sd_locked:
            print('training unet 11111111111111111111111')
            params += list(self.model.diffusion_model.input_blocks.parameters())
            params += list(self.model.diffusion_model.middle_block.parameters())
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        params += list(self.fusion_feature.parameters())
        params += list(self.fusion_latent.parameters())
        params += list(self.lgam.parameters())
        params += list(self.fusion_gate.parameters())
        params += list(self.fusion_feature_multi.parameters())
        params += list(self.fusion_latent_multi.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def forward(self, x, c, *args, **kwargs):
        if self.is_refine:
            t = torch.ones((x.shape[0],)).long().to(self.device) * self.used_timesteps - 1
        else:
            t = torch.randint(0, self.used_timesteps, (x.shape[0],)).long().to(self.device)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'
        c_latent = cond['c_latent'][0]
        if not self.is_refine:
            if not self.train_distortion_decoder_only:
                noise = default(noise, lambda : torch.randn_like(x_start)) + (c_latent - x_start) / self.lamba
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                model_output = self.apply_model(x_noisy, t, cond)
                if self.parameterization == 'x0':
                    target = x_start
                elif self.parameterization == 'eps':
                    target = x_start
                    model_output = self._predict_xstart_from_eps(x_noisy, t, model_output)
                elif self.parameterization == 'v':
                    target = self.get_v(x_start, noise, t)
                else:
                    raise NotImplementedError()
                loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
                loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
                logvar_t = self.logvar[t].to(self.device)
                loss = self.l_guide_weight * (loss_simple / torch.exp(logvar_t) + logvar_t).mean()
                p_attn = cond['pred_attn']
                g_attn = reference_confidence_target(cond['gt_attn'], p_attn, cond['gt_confidence'])
                loss_mask = F.mse_loss(p_attn, g_attn)
                loss_bpp = cond['bpp']
                guide_bpp = cond['q_bpp']
                loss_emb = cond['emb_loss']
                loss_guide = self.get_loss(c_latent, x_start)
                loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean(), f'{prefix}/q_bpp': guide_bpp.mean(), f'{prefix}/l_emb': loss_emb.mean(), f'{prefix}/l_guide': loss_guide.mean()})
                loss = loss + self.l_bpp_weight * (loss_bpp + loss_emb) + loss_mask * 0.1
                loss += self.l_guide_weight * loss_guide
            if self.train_distortion_decoder_only:
                noise = default(noise, lambda : torch.randn_like(c_latent))
                x_noisy = self.q_sample(x_start=c_latent, t=t, noise=noise)
                model_output = pytorch_checkpoint(self.apply_model, x_noisy, t, cond, use_reentrant=False)
                if self.parameterization == 'x0':
                    target = c_latent
                elif self.parameterization == 'eps':
                    target = noise
                else:
                    raise NotImplementedError()
                loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
                loss_bpp = cond['bpp'].mean()
                loss = self.l_guide_weight * loss_simple.mean() + self.l_bpp_weight * loss_bpp
                loss_dict.update({f'{prefix}/l_simple': loss_simple.mean(), f'{prefix}/l_bpp': loss_bpp, f'{prefix}/loss': loss})
                return (loss, loss_dict)
        else:
            (b, c, h, w) = c_latent.shape
            shape = (b, self.channels, h, w)
            noise = default(noise, lambda : torch.randn_like(c_latent))
            x_T = self.q_sample(x_start=c_latent, t=t, noise=noise)
            steps = self.fixed_step
            samples = self.sampler.sample_grad(steps, shape, cond, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=x_T)
            model_output = self.decode_first_stage_with_grad(samples)
            target = cond['target']
            loss_simple = self.get_loss(samples, c_latent, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})
            loss = self.l_guide_weight * loss_simple.mean()
            loss_mse = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/l_mse': loss_mse.mean()})
            loss = self.l_guide_weight * loss_mse.mean()
            loss_lpips = self.perceptual_loss(model_output, target)
            loss_dict.update({f'{prefix}/l_lpips': loss_lpips.mean()})
            loss += self.l_guide_weight * loss_lpips * 0.5
            loss_guide = self.get_loss(c_latent, x_start)
            loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
            loss += self.l_guide_weight * loss_guide
            loss_dict.update({f'{prefix}/loss': loss})
            loss_bpp = cond['bpp']
            guide_bpp = cond['q_bpp']
            loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
            loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
            loss += self.l_bpp_weight * loss_bpp
            loss_emb = cond['emb_loss']
            loss_dict.update({f'{prefix}/l_emb': loss_emb.mean()})
            loss += self.l_bpp_weight * loss_emb
        return (loss, loss_dict)

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]['p']
            val = self.ucg_training[k]['val']
            if val is None:
                val = ''
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val
        (loss, loss_dict) = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('global_step', self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = []
        (log, bpp) = self.log_images(batch, bs=None)
        out.append(bpp.cpu())
        save_dir = os.path.join(self.logger.save_dir, 'validation', f'{self.global_step}')
        os.makedirs(save_dir, exist_ok=True)
        image = log['samples'].detach().cpu()
        image = image.numpy().squeeze().transpose(1, 2, 0)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f'{batch_idx}.png')
        Image.fromarray(image).save(path)
        target = log['target'].detach().cpu()
        target = target.numpy().squeeze().transpose(1, 2, 0)
        target = (target * 255).clip(0, 255).astype(np.uint8)
        metric_data = [img2tensor(image).unsqueeze(0) / 255.0, img2tensor(target).unsqueeze(0) / 255.0]
        for (name, _) in self.calculate_metrics.items():
            out.append(self.metric_funcs[name](*metric_data))
        return out

    def on_validation_epoch_start(self):
        self.preprocess_model.quantize.reset_usage()
        return super().on_validation_epoch_start()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        outputs = np.array(outputs)
        avg_out = sum(outputs) / len(outputs)
        self.log('avg_bpp', avg_out[0], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        usage = self.preprocess_model.quantize.get_usage()
        self.log('usage', usage, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        for (i, (name, _)) in enumerate(self.calculate_metrics.items()):
            self.log(f'avg_{name}', avg_out[i + 1], prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def load_preprocess_ckpt(self, ckpt_path_pre):
        ckpt = torch.load(ckpt_path_pre)
        self.preprocess_model.load_state_dict(ckpt)
        print(['CONTROL WEIGHTS LOADED'])

    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        ckpt_base = torch.load(path)
        if synch_control:
            for key in list(ckpt_base['state_dict'].keys()):
                if 'diffusion_model.' in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=dim)[:control_dim]
                            else:
                                dim = 0
                                control_dim_0 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                dim = 1
                                control_dim_1 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([ckpt_base['state_dict'][key], ckpt_base['state_dict'][key]], dim=dim)[:control_dim_0, :control_dim_1, ...]
                        else:
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')
