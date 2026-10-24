import csv
import hashlib
import json
from pathlib import Path
from PIL import Image
SPLITS = ('train', 'val', 'test')
DATASETS = ('KITTI General', 'KITTI Stereo', 'Cityscapes', 'InStereo2K')

def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda : handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()

def read_manifests(paths):
    if set(paths) != set(SPLITS):
        raise ValueError('Exactly train, val and test manifests are required.')
    galleries = {}
    (seen_paths, seen_pixels, seen_scenes) = ({}, {}, {})
    for split in SPLITS:
        source = Path(paths[split]).resolve(strict=True)
        records = []
        with source.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            required = {'path', 'dataset', 'split', 'scene_id'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{source}: missing columns {sorted(required)}')
            for row in reader:
                if row['split'] != split or row['dataset'] not in DATASETS or (not row['scene_id'].strip()):
                    raise ValueError(f'{source}: invalid split, dataset or scene_id')
                image_path = Path(row['path'])
                if not image_path.is_absolute():
                    image_path = source.parent / image_path
                image_path = image_path.resolve(strict=True)
                with Image.open(image_path) as image:
                    rgb = image.convert('RGB')
                    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                scene = ('KITTI' if row['dataset'].startswith('KITTI') else row['dataset'], row['scene_id'])
                for (lookup, identity) in ((seen_paths, str(image_path)), (seen_pixels, pixels)):
                    if identity in lookup:
                        raise ValueError(f'Duplicate reference in {lookup[identity]} and {split}: {image_path}')
                    lookup[identity] = split
                if scene in seen_scenes and seen_scenes[scene] != split:
                    raise ValueError(f'Scene overlaps splits: {scene}')
                seen_scenes[scene] = split
                records.append(dict(path=str(image_path), dataset=row['dataset'], split=split, scene_id=row['scene_id'], sha256=digest(image_path), pixel_sha256=pixels))
        if not records:
            raise ValueError(f'Empty {split} gallery')
        galleries[split] = records
    return galleries

def load_gallery(root, split):
    import faiss
    import numpy as np
    if split not in SPLITS:
        raise ValueError(f'Invalid gallery split: {split}')
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['galleries']) != set(SPLITS):
        raise ValueError('Incomplete gallery bundle')
    info = manifest['galleries'][split]
    folder = root / split
    for name in ('gallery.index', 'metadata.npy', 'records.json'):
        if digest(folder / name) != info['sha256'][name]:
            raise ValueError(f'Gallery integrity check failed: {split}/{name}')
    records = json.loads((folder / 'records.json').read_text(encoding='utf-8'))
    metadata = np.load(folder / 'metadata.npy', allow_pickle=False)
    index = faiss.deserialize_index(np.frombuffer((folder / 'gallery.index').read_bytes(), dtype=np.uint8).copy())
    if index.ntotal != len(records) or len(metadata) != len(records) or len(records) != info['count']:
        raise ValueError('Gallery index and metadata counts disagree')
    if any((r['split'] != split for r in records)) or metadata.tolist() != [r['path'] for r in records]:
        raise ValueError('Gallery split or metadata mismatch')
    if not isinstance(index, faiss.IndexIDMap2) or not np.array_equal(faiss.vector_to_array(index.id_map), np.arange(len(records))):
        raise ValueError('Gallery requires contiguous reference IDs')
    return (index, metadata)
