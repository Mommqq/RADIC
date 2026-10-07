import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from ldm.util import default

from .compression import (
    Compression,
    ckbd_anchor,
    ckbd_merge,
    ckbd_nonanchor,
    ckbd_split,
    quantize_ste,
)
from .rdeic import RDEIC


class CompressionNoRateMask(Compression):
    """Compression module with rate-mask latent modulation removed."""

    def forward(self, x, h_y):
        y = self.encoder(x)
        h_y = self.encoder_y(h_y)
        z = self.hyper_enc(y)
        z_q, emb_loss, _ = self.quantize(z)

        hyper_params = self.hyper_dec(z_q)
        y_slices = [
            y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...]
            for i in range(len(self.slice_ch))
        ]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](
                    torch.cat([channel_ctx, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor

                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)

            _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
            _, q_slice_likelihoods = self.gaussian_conditional(
                y_slice, scales_slice, means_slice, False
            )
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
        no_rate_map = torch.zeros(
            y.shape[0], 1, y.shape[2], y.shape[3],
            device=y.device, dtype=y.dtype
        )
        return output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y, no_rate_map


class StrictAblationMixin:
    use_refs = False
    use_lgam = False
    use_ste = False

    def get_input(self, batch, k, bs=None, vis_path=None, *args, **kwargs):
        target, z, h, c = super(RDEIC, self).get_input(
            batch, self.first_stage_key, bs=bs, *args, **kwargs
        )
        c_latent, likelihoods, q_likelihoods, emb_loss, guide_hint, h_y, _ = self.preprocess_model(h, h)

        if self.use_refs and self.n_refs > 0:
            c_latent, guide_hint = self._fuse_references(batch, c_latent, guide_hint)

        num_pixels = z.shape[0] * z.shape[2] * z.shape[3] * 64
        bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in likelihoods)
        q_bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in q_likelihoods)

        return z, dict(
            c_crossattn=[c],
            c_latent=[c_latent],
            bpp=bpp,
            q_bpp=q_bpp,
            emb_loss=emb_loss,
            guide_hint=guide_hint,
            target=target,
        )

    def _fuse_references(self, batch, c_latent, guide_hint):
        with torch.no_grad():
            _, h_main_fixed = self.encode_first_stage(batch["main"].to(self.device))
            query_feat_fixed = F.normalize(torch.mean(h_main_fixed, dim=(2, 3)), p=2, dim=1)

        si_imgs, si_depths = self._load_reference_batch(batch, query_feat_fixed)
        total_n = si_imgs.shape[1]
        if total_n == 0:
            return c_latent, guide_hint

        f_refs = []
        z_refs = []
        for i in range(total_n):
            img_i = si_imgs[:, i]
            with torch.set_grad_enabled(self.training):
                _, h_i = self.encode_first_stage(img_i)
                f_yi, z_yi = self.preprocess_model.side_encode(h_i)

            if self.use_lgam:
                depth_i = si_depths[:, i]
                f_yi, z_yi, _, _ = self.lgam(
                    depth_i, guide_hint, f_yi, c_latent, z_yi, return_vis=False
                )

            f_refs.append(f_yi)
            z_refs.append(z_yi)

        f_stack = torch.stack(f_refs, dim=1)
        z_stack = torch.stack(z_refs, dim=1)

        if self.use_ste:
            # C4: LGAM references, STE fusion
            f_fused, _, _ = self.fusion_feature_multi(guide_hint, f_stack)
            z_fused, _, _ = self.fusion_latent_multi(c_latent, z_stack)
        elif self.use_lgam:
            # C3: LGAM references, simple average fusion
            f_fused = f_stack.mean(dim=1)
            z_fused = z_stack.mean(dim=1)
        else:
            # C2: identity references, simple average fusion
            f_fused = f_stack.mean(dim=1)
            z_fused = z_stack.mean(dim=1)

        guide_hint = guide_hint + self.fusion_feature(torch.cat([guide_hint, f_fused], dim=1))
        cat_z = torch.cat([c_latent, z_fused], dim=1)
        c_latent = self.fusion_gate(cat_z) * c_latent + self.fusion_latent(cat_z)
        return c_latent, guide_hint

    def _load_reference_batch(self, batch, query_feat_fixed):
        if not self.training and "multi_side" in batch:
            si_imgs = batch["multi_side"].to(self.device)
            if self.use_lgam:
                b, k, c, h, w = si_imgs.shape
                depths = self._get_depth_maps(si_imgs.view(b * k, c, h, w))
                si_depths = depths.view(b, k, 1, h, w)
            else:
                b, k, _, h, w = si_imgs.shape
                si_depths = torch.zeros(b, k, 1, h, w, device=self.device, dtype=si_imgs.dtype)
            return si_imgs, si_depths

        q_f_fixed = query_feat_fixed.detach().cpu().numpy().astype("float32")
        _, indices = self.index.search(q_f_fixed, self.n_refs)
        si_imgs, si_depths = self.fetch_retrieved_data(indices.astype("int64"))
        return si_imgs, si_depths

    def p_losses(self, x_start, cond, t, noise=None):
        if self.is_refine or self.train_distortion_decoder_only:
            raise NotImplementedError("Strict ablations support the first-stage RDEIC training path only.")

        loss_dict = {}
        prefix = "T" if self.training else "V"
        c_latent = cond["c_latent"][0]

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
        loss_dict.update({f"{prefix}/l_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = self.l_guide_weight * (loss_simple / torch.exp(logvar_t) + logvar_t).mean()

        loss_bpp = cond["bpp"]
        guide_bpp = cond["q_bpp"]
        loss_emb = cond["emb_loss"]
        loss_guide = self.get_loss(c_latent, x_start)

        loss_dict.update({
            f"{prefix}/l_bpp": loss_bpp.mean(),
            f"{prefix}/q_bpp": guide_bpp.mean(),
            f"{prefix}/l_emb": loss_emb.mean(),
            f"{prefix}/l_guide": loss_guide.mean(),
        })

        loss = loss + self.l_bpp_weight * (loss_bpp + loss_emb)
        loss = loss + self.l_guide_weight * loss_guide
        loss_dict.update({f"{prefix}/loss": loss.mean()})
        return loss, loss_dict


class RDEIC_C1_Strict(StrictAblationMixin, RDEIC):
    use_refs = False
    use_lgam = False
    use_ste = False


class RDEIC_C2_Strict(StrictAblationMixin, RDEIC):
    use_refs = True
    use_lgam = False
    use_ste = False


class RDEIC_C3_Strict(StrictAblationMixin, RDEIC):
    use_refs = True
    use_lgam = True
    use_ste = False


class RDEIC_C4_Strict(StrictAblationMixin, RDEIC):
    use_refs = True
    use_lgam = True
    use_ste = True
