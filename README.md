# StyleGAN for Autolume

This branch (`autolume`) adapts [NVlabs StyleGAN3](https://github.com/NVlabs/stylegan3) for use in [Autolume](https://github.com/Metacreation-Lab/autolume), which builds on it for both model training and real-time inference. It stays close to upstream: changes are limited to packaging, Apple Silicon (MPS) support, the custom-ops build system and training quality of life. The `main` branch tracks NVlabs upstream, and the original StyleGAN3 documentation applies to everything not listed below.

## Added features

### Packaging

- uv-managed project: Python 3.12, `torch 2.8.0` (cu128 index on Windows/Linux, PyPI on macOS), numpy 2.

### Custom-ops build system

- Shared build cache keyed by source digest, Python version, torch version and compute capabilities: GPUs of any covered capability share builds, and a warmed cache loads without a compiler, CUDA toolkit or ninja. Failed builds are detected and cleaned automatically. Set `TORCH_EXTENSIONS_DIR` to relocate the cache.
- `scripts/precompile_ops.py` warms the same cache ahead of time, for the current device by default or an explicit `--arch` list, building each op once as a single fatbin covering all requested capabilities.
- Cache entries record build provenance (torch, CUDA runtime, CUDA toolkit); builds warn on a toolkit/runtime major-version mismatch and precompilation refuses it unless `--bypass-matching-toolkit` is passed.
- CUDA 13 compatibility fix in `filtered_lrelu`.

### Apple Silicon (MPS)

- `--device` option on `gen_images.py`, `gen_video.py` and `train.py`, defaulting to auto (`cuda` > `mps` > `cpu`). Ops without MPS kernels fall back to CPU.
- Training runs on MPS and CPU (single process, `--gpus=1`).
- The visualizer works on macOS: fixed-function imgui backend for the GL 2.1 context, 1:1 framebuffer, monitor work-area crash guard, and a working async renderer.
- Mouse presses are latched so short clicks register at low frame rates (all platforms).
- Texture mipmaps disabled on macOS (GL 3.0+ requirement).

### Training

- `resume_kimg` is derived from the snapshot filename, so resumed trainings continue their kimg count and snapshot numbering.
- `--augpipe` presets from StyleGAN2-ADA (`blit`, `geom`, `color`, …, `bgc` default).
- Opt-in energy and emissions tracking via codecarbon (`--track-emissions`, off by default).

### Dataset tool

- Palette (P-mode) images are converted to RGB instead of crashing the pipeline.

## Quick start

```bash
git clone -b autolume git@github.com:Metacreation-Lab/stylegan3.git
cd stylegan3
uv sync
uv run python gen_images.py --network=<model.pkl> --seeds=0-3 --outdir=out
```

Optional, to ship or share prebuilt custom ops:

```bash
uv run python scripts/precompile_ops.py            # current GPU
uv run python scripts/precompile_ops.py --arch=7.5,8.6,8.9,12.0
```

## License

NVIDIA Source Code License, unchanged from upstream — see [LICENSE.txt](LICENSE.txt).
