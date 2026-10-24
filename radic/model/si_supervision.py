import torch
import torch.nn.functional as F

def reference_confidence_target(soft_attention, prediction, confidence):
    if soft_attention.ndim != 4 or prediction.ndim != 4:
        raise ValueError('Attention and prediction must be 4D tensors.')
    if soft_attention.shape[0] != prediction.shape[0] or prediction.shape[1] != 1:
        raise ValueError('Batch sizes must agree and prediction must have one channel.')
    with torch.no_grad():
        if soft_attention.shape[1] == 0:
            return torch.zeros_like(prediction)
        if confidence.shape != soft_attention.shape:
            raise ValueError('Confidence and attention shapes must agree.')
        target = (soft_attention.detach() * confidence.detach()).amax(dim=1, keepdim=True)
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(target, size=prediction.shape[-2:], mode='bilinear', align_corners=False)
        return target

def masked_reference_attention(scores, valid):
    mask = valid[:, :, None, None].to(device=scores.device, dtype=torch.bool)
    if scores.shape[1] == 0:
        return scores
    weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(1)
    weights = weights * mask
    return weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)

def hyper_index_bpp(shape, codebook_size, num_pixels):
    import math
    if codebook_size < 2 or num_pixels <= 0 or len(shape) != 4:
        raise ValueError('Invalid hyperprior rate dimensions.')
    return shape[0] * shape[-2] * shape[-1] * math.log2(codebook_size) / num_pixels
