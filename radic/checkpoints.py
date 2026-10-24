import math
import os
from pathlib import Path
import torch

def save_training_checkpoint(folder, model, optimizer, scheduler, epoch, step, best_psnr, metrics=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {key: float(value) for (key, value) in metrics.items()} if metrics is not None else None
    score = metrics['psnr'] if metrics is not None else float('nan')
    improved = math.isfinite(score) and score > best_psnr
    if improved:
        best_psnr = score
    state = dict(epoch=epoch, global_step=step, state_dict=model.state_dict(), best_psnr=best_psnr, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), metrics=metrics)
    for name in ('last.pth', 'best_psnr.pth') if improved else ('last.pth',):
        temporary = folder / (name + '.tmp')
        torch.save(state, temporary)
        os.replace(temporary, folder / name)
    return best_psnr
