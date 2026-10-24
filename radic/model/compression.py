import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils1.func import get_scale_table
from model.compression_modules import *

class EncoderPredictor(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels // 4, 3, 1, 1), nn.SiLU(), nn.Conv2d(channels // 4, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, y):
        return self.net(y)

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
        self.proj_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.local_context = nn.ModuleList((nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2) for i in range(len(slice_ch))))
        self.channel_context = nn.ModuleList((ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2) if i else None for i in range(slice_num)))
        self.entropy_parameters_anchor = nn.ModuleList((EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2) if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2) for i in range(slice_num)))
        self.entropy_parameters_nonanchor = nn.ModuleList((EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2) if i else EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2) for i in range(slice_num)))
        self.codebook_size = codebook_size
        self.quantize = VectorQuantiser(self.codebook_size, N, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)
        self.predictor = EncoderPredictor(M)

    def get_embedding(self, feat):
        emb = self.proj_head(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def side_encode(self, h_i):
        f_y = self.encoder_y(h_i)
        f_y = self.decoder_y(f_y)
        z_y = self.out_y(f_y)
        return (f_y, z_y)

    def forward(self, x, h_y):
        y = self.encoder(x)
        h_y = self.encoder_y(h_y)
        z = self.hyper_enc(y)
        self.hyper_index_shape = tuple(z.shape)
        (z_q, emb_loss, _) = self.quantize(z)
        pred_attn = self.predictor(y)
        attenuation = 1.0 - 0.7 * pred_attn
        y = y * attenuation
        hyper_params = self.hyper_dec(z_q)
        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:i + 1]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []
        for (idx, y_slice) in enumerate(y_slices):
            (slice_anchor, slice_nonanchor) = ckbd_split(y_slice)
            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                (scales_anchor, means_anchor) = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                (scales_nonanchor, means_nonanchor) = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                (_, y_slice_likelihoods) = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                (_, q_slice_likelihoods) = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                (scales_anchor, means_anchor) = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                (scales_nonanchor, means_nonanchor) = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                (_, y_slice_likelihoods) = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                (_, q_slice_likelihoods) = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
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
        output_y = self.out(h_y)
        return (output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y, pred_attn)

    def compress(self, x):
        y = self.encoder(x)
        z = self.hyper_enc(y)
        (z_q, encoding_indices) = self.quantize.quant(z)
        y = y * (1.0 - 0.7 * self.predictor(y))
        torch.backends.cudnn.deterministic = True
        z_strings = compress_hyper_latent(encoding_indices, self.codebook_size)
        hyper_params = self.hyper_dec(z_q)
        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:i + 1]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        for (idx, y_slice) in enumerate(y_slices):
            (slice_anchor, slice_nonanchor) = ckbd_split(y_slice)
            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                (scales_anchor, means_anchor) = params_anchor.chunk(2, 1)
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                (scales_nonanchor, means_nonanchor) = params_nonanchor.chunk(2, 1)
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                (scales_anchor, means_anchor) = params_anchor.chunk(2, 1)
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                (scales_nonanchor, means_nonanchor) = params_nonanchor.chunk(2, 1)
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.backends.cudnn.deterministic = False
        return {'strings': [y_strings, [z_strings]], 'shape': z.size()[-2:]}

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
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                (scales_anchor, means_anchor) = params_anchor.chunk(2, 1)
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                (scales_nonanchor, means_nonanchor) = params_nonanchor.chunk(2, 1)
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                (scales_anchor, means_anchor) = params_anchor.chunk(2, 1)
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                (scales_nonanchor, means_nonanchor) = params_nonanchor.chunk(2, 1)
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)
        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False
        guide_hint = self.decoder(y_hat)
        output = self.out(guide_hint)
        return (output, guide_hint)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
