# UnReflectAnything
### RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision
[![Project](https://img.shields.io/badge/Project-webpage-543fce?logo=")](https://alberto-rota.github.io/UnReflectAnything/)
[![PyPI](https://img.shields.io/badge/pip%20install-PyPI-3776AB?logo=python&logoColor=3776AB)](https://pypi.org/project/unreflectanything/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=B31B1B)](https://huggingface.co/spaces/Stable-X/StableDelight)
[![Weights](https://img.shields.io/badge/Weights-Hugging%20Face%20-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/spaces/AlbeRota/UnReflectAnything)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face%20-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/AlbeRota/UnReflectAnything)
[![License](https://img.shields.io/badge/License-MIT-1E811F)](https://mit-license.org/)


## Install from PyPI

```bash
pip install unreflectanything
```

For GPU support, install PyTorch with the appropriate CUDA version for your system (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).

---

## Downloading weights

Pretrained weights are not included in the package. After installing, download them once:

```bash
unreflectanything download-weights
```

Weights are stored by default in `~/.cache/unreflectanything/weights` (or `$XDG_CACHE_HOME/unreflectanything/weights` if set). Use `--output-dir` to choose another location. To use a custom weights repository (e.g. on Hugging Face), set the environment variable `UNREFLECTANYTHING_WEIGHTS_REPO`.

For the download command to work, install the optional dependency: `pip install unreflectanything[weights]`.

---

## CLI usage

After installation you get the `unreflectanything` command (aliases: `unreflect`, `ura`; all three do the same thing). For development, use `pip install -e .` so the same entry points work from the repo.

| Subcommand | Description |
|------------|-------------|
| `train` | Run training (e.g. `unreflectanything train --config config_train.yaml`) |
| `test` | Run evaluation (`unreflectanything test --config config_test.yaml`) |
| `inference` | Run inference on an image directory (`unreflectanything inference --config config_inference.yaml`) |
| `download-weights` | Download pretrained weights (see above) |
| `sweep` | Launch a Weights & Biases sweep |
| `agent` | Run a W&B sweep agent |
| `completion` | Print shell completion (bash/zsh): `source <(unreflectanything completion bash)` |

Inference expects a YAML config with `weights_path`, `input_dir`, `output_dir`, and other options (see `config_inference.yaml`). You can point `weights_path` to the cache directory after running `download-weights`.

---

## Python API

Use the package programmatically:

```python
import unreflectanything as ura

# Run training or testing
ura.run_pipeline(mode="train")   # or mode="test"

# Run inference from options
from pathlib import Path
from unreflectanything import InferenceOptions, run_inference

options = InferenceOptions(
    weights_path=Path("path/to/full_model_weights.pt"),
    input_dir=Path("input/images"),
    output_dir=Path("output/diffuse"),
)
ura.run_inference(options)

# Utility: compute highlight mask from RGB batch [B,3,H,W]
import torch
mask = ura.compute_highlight_mask(rgb_batch, threshold=0.7)  # [B,1,H,W]

# Default weights cache directory
cache_dir = ura.get_weights_cache_dir()
```

---

## Testing the package

To verify the package works before or after publishing:

- **Local**: Run `uv build`, then install the wheel from `dist/` into a venv and run `unreflectanything --help` and `import unreflectanything as ura`.
- **TestPyPI**: Run `python mgmt/upload_to_pypi.py testpypi` (or `python -m mgmt.upload_to_pypi testpypi`), then in a new venv install with `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ unreflectanything`.

See [TESTING.md](TESTING.md) for step-by-step commands.
