# UnReflectAnything

[![Project](https://img.shields.io/badge/Project-webpage-543fce?logo=")](https://alberto-rota.github.io/UnReflectAnything/)
[![PyPI](https://img.shields.io/badge/pip%20install-PyPI-3776AB?logo=python&logoColor=3776AB)](https://pypi.org/project/unreflectanything/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=B31B1B)](https://huggingface.co/spaces/Stable-X/StableDelight)
[![Weights](https://img.shields.io/badge/Weights-HuggingFace%20-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/spaces/AlbeRota/UnReflectAnything)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/AlbeRota/UnReflectAnything)
[![License](https://img.shields.io/badge/License-MIT-1E811F)](https://mit-license.org/)
### RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision
UnReflectAnything inputs any RGB image and removes specular highlights, returning a clean diffuse-only outputs. We trained UnReflectAnything by synthetizing specularities and supervising in DINOv3 feature space.


UnReflectAnything works on both natural indoor and surgical/endoscopic domain data. 

---
![examples](https://github.com/alberto-rota/UnReflectAnything/blob/main/assets/header.png)

## Installation
```bash
pip install unreflectanything
```
Install UnReflectAnything as a Python Package. 

The minimum required Python version is 3.11, but development and all experiments have been bases on **Python 3.12**.

For GPU support, make sure PyTorch comes with CUDA version for your system (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).


## Setting up
After pip-installing, you can use the `unreflectanything` CLI command, which is also aliased to `unreflect` and `ura`. The three commands are equivalent.

With the CLI you can already download the model weights with
```bash
unreflectanything download --weights
```
and some sample images with 
```bash
unreflectanything download --images
```

Weights are stored by default in `~/.cache/unreflectanything/weights` (or `$XDG_CACHE_HOME/unreflectanything/weights` if set ; `%LOCALAPPDATA%\unreflectanything` for Windows). Use `--output-dir` to choose another location. 

Both the weights and images are stored on the [HuggingFace Model Repo](https://huggingface.co/spaces/AlbeRota/UnReflectAnything).

## Enable shell completion
Shell completion is available for the `bash` and `zsh` shells. Run
```bash
unreflectanything completion bash
```
and execute the `echo ...` command that gets printed.


## Command Line Interface
Get an overview of the available CLI endpoints with 
```
unreflectanything --help   # alias 'unreflect --help' alias 'ura --help'
```
Refer to the [Wiki](https://github.com/alberto-rota/UnReflectAnything/wiki) to get detailed documentation about each endpoint. We report a summary of the available subcommands. Remember that `ura` is aliased to the `unreflectanything` command

| Subcommand | Description | Command |
|------------|-------------|-------------|
| `inference` | Run inference on an image directory  |`ura inference --input /path/to/images --output /path/to/unref_images` |
| `train` | Run training | `ura train --config config_train.yaml`|
| `test` | Run evaluation on a trained model |`ura test --config config_test.yaml`|
| `download` | Download checkpoint weights, sample images, notebooks |`ura download --weights`|
| `verify` | Verify weights installation and compatibility, as well as dataset directory structure | `ura verify --dataset /path/to/dataset`|
| `evaluate` | Compute metrics on output data | `ura evaluate --output /path/to/unref_images --gt /path/to/groundtruth_images/`|
| `completion` | Print shell completion (bash/zsh): |`ura completion bash` |
| `cite` | Print shell completion (bash/zsh)| `ura cite --bibtex` |

## Python API

The same endpoints above are exposed as a Python API. Refer to the [Wiki](https://github.com/alberto-rota/UnReflectAnything/wiki) to get detailed documentation about each endpoint. A few examples are reported below

```python
import unreflectanything
from unreflectanything import UnReflectModel
import torch

img = torch.from_numpy(np.array(PIL.Image.open("path/to/image.jpg")))

# Instantiate and call the model
unreflect = UnReflectModel()
unreflected_img = unreflect(img)

# Run training or testing
unreflectanything.run_pipeline(mode="train")   # or mode="test"
unreflectanything.run_pipeline(mode="test")   

# Run inference from options
options = unreflectanything.InferenceOptions(
    weights_path="path/to/full_model_weights.pt",
    input_dir="path/to/input/images",
    output_dir="path/to/output/diffuse",
)
unreflectanything.inference(options)

```

## Citation
If you include UnReflectAnything in your pipline or research work, please cite us  
```
@misc{rota2025unreflectanythingrgbonlyhighlightremoval,
      title={UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision}, 
      author={Alberto Rota and Mert Kiray and Mert Asim Karaoglu and Patrick Ruhkamp and Elena De Momi and Nassir Navab and Benjamin Busam},
      year={2025},
      eprint={2512.09583},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.09583}, 
}
```
