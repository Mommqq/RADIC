# RADIC

Official research code and project page for **Retrieval-Augmented Distributed Image Compression with Multi-Reference Side Information**.

[Project page](https://mommqq.github.io/RADIC/) | [Paper](https://mommqq.github.io/RADIC/assets/RADIC.pdf)

RADIC retrieves multiple references from a decoder-side visual gallery, aligns them with depth-aware geometry, and selectively aggregates useful content for extremely low-bitrate reconstruction.

## Repository layout

- `docs/`: static project page used by GitHub Pages.
- `radic/`: training, inference, retrieval, model, configuration, and dataset code.

## Quick start

See [radic/README.md](radic/README.md) for environment setup, expected data layout, gallery construction, training, and evaluation commands.

## Citation

```bibtex
@article{xu2026radic,
  title   = {Retrieval-Augmented Distributed Image Compression with Multi-Reference Side Information},
  author  = {Xu, Guojun and Xiang, Jianwen and Zhang, Mingyang and Xie, Yaning and Zhou, Junwei},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

RADIC-specific code is released under the BSD 3-Clause License. Third-party components retain their original licenses and attribution requirements.
