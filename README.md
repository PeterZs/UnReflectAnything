<div align="center">

<p>
<b>Alberto Rota<sup>1</sup>, Mert Kiray<sup>2,3</sup>, Mert Asim Karaoglu<sup>4,2</sup>, Patrick Ruhkamp<sup>2</sup>,<br>
Elena De Momi<sup>1</sup>, Nassir Navab<sup>2,3</sup>, Benjamin Busam<sup>2,3</sup></b><br>

<sup>1</sup>Politecnico di Milano &nbsp;&nbsp; <sup>2</sup>Technical University of Munich &nbsp;&nbsp; <sup >3</sup>Munich Center for Machine Learning (MCML) &nbsp;&nbsp; <sup>4</sup>ImFusion
</p> 

https://huggingface.co/AlbeRota/UnReflectAnything 

<p>
<a href="#" style="padding:12px 28px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;border-radius:50px;text-decoration:none;margin:8px;display:inline-block;font-weight:600;box-shadow:0 4px 15px rgba(102, 126, 234, 0.4);transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);position:relative;overflow:hidden;" onmouseover="this.style.transform='translateY(-4px) scale(1.05)';this.style.boxShadow='0 8px 25px rgba(102, 126, 234, 0.6)';this.style.background='linear-gradient(135deg, #764ba2 0%, #667eea 100%)'" onmouseout="this.style.transform='translateY(0) scale(1)';this.style.boxShadow='0 4px 15px rgba(102, 126, 234, 0.4)';this.style.background='linear-gradient(135deg, #667eea 0%, #764ba2 100%)'">📄 Paper</a> 
<a href="https://github.com/alberto-rota/UnReflectAnything" style="padding:12px 28px;background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%);color:#fff;border-radius:50px;text-decoration:none;margin:8px;display:inline-block;font-weight:600;box-shadow:0 4px 15px rgba(17, 153, 142, 0.4);transition:all 0.3s cubic-bezier(0.4, 0, 0.2, 1);position:relative;overflow:hidden;" onmouseover="this.style.transform='translateY(-4px) scale(1.05)';this.style.boxShadow='0 8px 25px rgba(17, 153, 142, 0.6)';this.style.background='linear-gradient(135deg, #38ef7d 0%, #11998e 100%)'" onmouseout="this.style.transform='translateY(0) scale(1)';this.style.boxShadow='0 4px 15px rgba(17, 153, 142, 0.4)';this.style.background='linear-gradient(135deg, #11998e 0%, #38ef7d 100%)'">💻 Code</a>
</p> 

<img src="assets/header.png" alt="method overview" width="90%"/>

</div>

---

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
    weights_path=Path("path/to/weights_best.pt"),
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
