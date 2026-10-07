# RADIC research code

This directory contains the model, training, inference, retrieval, configuration, and dataset-loading code used by RADIC.

## Environment

The reference experiments use Python 3.10 and CUDA-capable PyTorch.

```bash
conda create -n radic python=3.10 -y
conda activate radic
pip install -r requirements_redic.txt
```

Download the Stable Diffusion 2.1 base checkpoint and place it at:

```text
weight/lpips/v2-1_768-ema-pruned.ckpt
```

Depth maps are generated with Depth Anything V3. Install it from its official repository and keep generated maps outside this repository.

## Data layout

The loaders expect independently configurable dataset roots. A typical KITTI layout is:

```text
datasets/
  KITTI/
    train/
    val/
    test/
```

Dataset files, depth maps, FAISS indexes, and checkpoints are intentionally excluded from Git.

## Build the decoder-side gallery

Update the dataset and output paths in `extract_gallery.py`, then run:

```bash
python extract_gallery.py
```

The model configuration expects the resulting index and metadata at:

```text
retrieval_db/gallery.index
retrieval_db/metadata.npy
```

## Train

```bash
torchrun --nproc_per_node=4 train.py \
  --train_config configs/train_rdeic.yaml \
  --model_config configs/model/rdeic.yaml \
  --data_root /path/to/KITTI \
  --save_dir outputs/radic
```

The operating point is controlled by the loss weights in the model configuration. The released configuration uses three retrieved references and two diffusion sampling steps.

## Evaluate

Set the dataset root, checkpoint, gallery index, and gallery metadata in the evaluation configuration, then run:

```bash
python test.py --help
```

Evaluation should report both estimated bitrate and reconstruction metrics. The paper uses LPIPS, DISTS, FID, KID, NIQE, PSNR, and MS-SSIM on KITTI General, KITTI Stereo, Cityscapes, and InStereo2K.

## Reproducibility scope

- Training domain: KITTI General.
- Input resolution: 256 × 512 after center crop and resize.
- Retrieved references: `K=3` during training and the main evaluation.
- Diffusion decoding: two sampling steps.
- Decoder-side gallery files are derived data and are not committed.

Pretrained checkpoints and ready-to-use gallery indexes are large artifacts and will be distributed separately.
