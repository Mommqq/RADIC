import argparse
import json
import uuid
from pathlib import Path
from gallery import SPLITS, digest, read_manifests

def build_galleries(config_path, checkpoint_path, manifests, output, device):
    galleries = read_manifests(manifests)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    import faiss
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from torchvision.transforms import functional as TF, InterpolationMode
    from utils1.utils import instantiate_from_config
    from tqdm import tqdm
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config).to(device).eval()
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state.get('state_dict', state), strict=True)
    staging = destination.with_name(destination.name + '.building-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    bundle = dict(checkpoint_sha256=digest(checkpoint_path), config_sha256=digest(config_path), crop_hw=[370, 740], resize_hw=[256, 512], galleries={})
    with torch.no_grad():
        for split in SPLITS:
            records = galleries[split]
            folder = staging / split
            folder.mkdir()
            index = None
            for (i, record) in enumerate(tqdm(records, desc=split)):
                path = record['path']
                if digest(path) != record['sha256']:
                    raise ValueError(f'Image changed during indexing: {path}')
                with Image.open(path) as image:
                    image = TF.resize(TF.center_crop(image.convert('RGB'), (370, 740)), (256, 512), interpolation=InterpolationMode.BICUBIC)
                    tensor = TF.to_tensor(image).unsqueeze(0).to(device) * 2 - 1
                (_, latent) = model.encode_first_stage(tensor)
                latent = model.get_first_stage_encoding(latent)
                (feature, _) = model.preprocess_model.side_encode(latent)
                vector = model.preprocess_model.get_embedding(feature).float().cpu().numpy()
                if vector.ndim != 2 or vector.shape[0] != 1 or (not np.isfinite(vector).all()):
                    raise ValueError(f'Invalid gallery descriptor: {path}')
                if index is None:
                    index = faiss.IndexIDMap2(faiss.IndexFlatL2(vector.shape[1]))
                index.add_with_ids(vector.astype('float32'), np.asarray([i], dtype='int64'))
            (folder / 'gallery.index').write_bytes(faiss.serialize_index(index).tobytes())
            np.save(folder / 'metadata.npy', np.asarray([r['path'] for r in records]))
            (folder / 'records.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
            bundle['galleries'][split] = dict(count=len(records), sha256={name: digest(folder / name) for name in ('gallery.index', 'metadata.npy', 'records.json')})
    (staging / 'manifest.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    staging.rename(destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model/radic.yaml')
    parser.add_argument('--ckpt', required=True)
    for split in SPLITS:
        parser.add_argument(f'--{split}_manifest', required=True)
    parser.add_argument('--output', default='retrieval_db')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    build_galleries(args.config, args.ckpt, {s: getattr(args, s + '_manifest') for s in SPLITS}, args.output, args.device)
if __name__ == '__main__':
    main()
