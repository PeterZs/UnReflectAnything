"""Inference API for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from os import PathLike

import torch


def _config_to_path_or_none(
    config: Optional[Union[str, PathLike, Path, dict]],
) -> Optional[Path]:
    """Return a Path for model config when config is a path; else None (use model default)."""
    if config is None or isinstance(config, dict):
        return None
    p = Path(config).expanduser().resolve()
    return p if p.exists() else None


def inference(
    input: Union[str, PathLike, Path, torch.Tensor],
    output: Optional[Union[str, PathLike, Path]] = None,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    config: Optional[Union[str, PathLike, Path, dict]] = None,
    device: str = "cuda",
    batch_size: int = 4,
    brightness_threshold: float = 0.8,
    resize_output: bool = True,
    verbose: bool = False,
    model: Optional[Any] = None,
) -> Optional[torch.Tensor]:
    """Run inference on input image(s) to remove specular reflections.

    This function runs the UnReflectAnything model on input images to produce
    diffuse (reflection-free) outputs. It supports both file-based and tensor-based
    workflows.

    Args:
        input: Input source. Can be:
            - Path to a single image file
            - Path to a directory containing images
            - Tensor of shape [B, 3, H, W] (batch of RGB images, values in [0, 1])
        output: Output destination. If provided, results are saved to disk.
            If None, returns the result as a tensor.
        weights_path: Path to model weights. Ignored when `model` is provided.
            Defaults to the cache directory when model is None.
        config: Configuration source (YAML path or dict). Ignored when `model` is provided.
        device: Device to run inference on (e.g. 'cuda', 'cpu'). Ignored when `model` is provided.
        batch_size: Number of images to process per forward pass (default: 4).
        brightness_threshold: Threshold for highlight mask computation (0.0-1.0).
        resize_output: If True, resize output images to match original input dimensions.
        verbose: If True, print progress information.
        model: Optional pre-built model instance (e.g. from ``ura.model(pretrained=True)``).
            If provided, no model is created internally and weights_path/config/device are ignored.

    Returns:
        If output is None: Tensor of shape [B, 3, H, W] with diffuse predictions.
        If output is provided: None (results saved to disk).
    """
    from torch import Tensor

    from ._shared import _resolve_device

    is_tensor_input = isinstance(input, Tensor)

    if is_tensor_input:
        return _inference_tensor(
            input_tensor=input,
            model=model,
            weights_path=weights_path,
            config=config,
            device=device,
            brightness_threshold=brightness_threshold,
            verbose=verbose,
        )

    input_path = Path(input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if output is None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            return _inference_files_return_tensors(
                input_path=input_path,
                output_path=output_path,
                model=model,
                weights_path=weights_path,
                config=config,
                device=device,
                batch_size=batch_size,
                brightness_threshold=brightness_threshold,
                resize_output=resize_output,
                verbose=verbose,
            )

    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if model is not None:
        _inference_files_save(
            model=model,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            brightness_threshold=brightness_threshold,
            resize_output=resize_output,
            verbose=verbose,
        )
        return None

    from inference import InferenceOptions, run_inference as _run_inference_files

    from ._shared import (
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
        _apply_config_to_options,
    )

    resolved_weights = (
        get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
        if weights_path is None
        else Path(weights_path).expanduser().resolve()
    )
    if not resolved_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {resolved_weights}. "
            "Run 'unreflect download --weights' first."
        )

    options = InferenceOptions(
        weights_path=resolved_weights,
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        output_dir=output_path if output_path.is_dir() else output_path.parent,
        device=_resolve_device(device),
        batch_size=batch_size,
        brightness_threshold=brightness_threshold,
        resize_output=resize_output,
        monitor_usage=False,
    )
    if config is not None:
        options = _apply_config_to_options(options, config)
    _run_inference_files(options)
    return None


def _inference_tensor(
    input_tensor: torch.Tensor,
    model: Optional[Any] = None,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, dict]] = None,
    device: str = "cuda",
    brightness_threshold: float = 0.8,
    verbose: bool = False,
) -> torch.Tensor:
    """Run inference on a tensor input, returning a tensor output."""
    from ._shared import _resolve_device

    if input_tensor.dim() != 4:
        raise ValueError(
            f"Input tensor must be 4D [B,C,H,W], got {input_tensor.dim()}D"
        )
    if input_tensor.shape[1] != 3:
        raise ValueError(
            f"Input tensor must have 3 channels, got {input_tensor.shape[1]}"
        )

    if model is None:
        from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir
        from .model_ import model as model_factory

        resolved_weights = (
            get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
            if weights_path is None
            else Path(weights_path).expanduser().resolve()
        )
        if not resolved_weights.exists():
            raise FileNotFoundError(
                f"Weights not found at {resolved_weights}. "
                "Run 'unreflect download --weights' first."
            )
        config_path = _config_to_path_or_none(config)
        torch_device = torch.device(_resolve_device(device))
        model = model_factory(
            pretrained=True,
            weights_path=resolved_weights,
            config_path=config_path,
            device=torch_device,
            verbose=verbose,
        )
    else:
        torch_device = getattr(model, "device", None)
        if torch_device is None:
            torch_device = torch.device(_resolve_device(device))
        elif isinstance(torch_device, str):
            torch_device = torch.device(torch_device)

    input_tensor = input_tensor.to(device=torch_device, dtype=torch.float32)
    # [B, 1, H, W] highlight mask from brightness
    inpaint_mask = (input_tensor.mean(1, keepdim=True) > brightness_threshold).to(
        torch_device, dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        outputs = model(
            images=input_tensor,
            inpaint_mask_override=inpaint_mask,
            return_dict=True,
        )
    diffuse = outputs.get("diffuse")
    if diffuse is None:
        raise KeyError("Model output does not contain 'diffuse'")
    return diffuse.clamp(0.0, 1.0)


def _inference_files_return_tensors(
    input_path: Path,
    output_path: Path,
    model: Optional[Any] = None,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, dict]] = None,
    device: str = "cuda",
    batch_size: int = 4,
    brightness_threshold: float = 0.8,
    resize_output: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """Run file-based inference but return results as tensors instead of saving."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    from utilities.inference import list_image_paths

    from ._shared import (
        DEFAULT_IMAGE_EXTENSIONS,
        DEFAULT_WEIGHTS_FILENAME,
        get_cache_dir,
        _resolve_device,
    )

    if model is None:
        from .model_ import model as model_factory

        resolved_weights = (
            get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
            if weights_path is None
            else Path(weights_path).expanduser().resolve()
        )
        if not resolved_weights.exists():
            raise FileNotFoundError(
                f"Weights not found at {resolved_weights}. "
                "Run 'unreflect download --weights' first."
            )
        config_path = _config_to_path_or_none(config)
        torch_device = torch.device(_resolve_device(device))
        model = model_factory(
            pretrained=True,
            weights_path=resolved_weights,
            config_path=config_path,
            device=torch_device,
            verbose=verbose,
        )

    torch_device = getattr(model, "device", None)
    if torch_device is None:
        torch_device = torch.device(_resolve_device(device))
    elif isinstance(torch_device, str):
        torch_device = torch.device(torch_device)

    target_side = getattr(model, "image_size", None) or getattr(
        model._model.dinov3.config, "image_size", 448
    )
    target_size = (target_side, target_side)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list_image_paths(input_path, DEFAULT_IMAGE_EXTENSIONS)

    results = []
    model.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for path in batch_paths:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                tensor = TF.to_tensor(rgb_img)
                resized = TF.resize(tensor, target_size, antialias=True)
                batch_tensors.append(resized)

        rgb_batch = torch.stack(batch_tensors, dim=0).to(
            device=torch_device, dtype=torch.float32
        )
        inpaint_mask = (
            rgb_batch.mean(1, keepdim=True) > brightness_threshold
        ).to(device=torch_device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(
                images=rgb_batch,
                inpaint_mask_override=inpaint_mask,
                return_dict=True,
            )
        diffuse = outputs.get("diffuse")
        if diffuse is None:
            raise KeyError("Model output does not contain 'diffuse'")
        results.append(diffuse.clamp(0.0, 1.0).cpu())

    return torch.cat(results, dim=0)


def _inference_files_save(
    model: Any,
    input_path: Path,
    output_path: Path,
    batch_size: int = 4,
    brightness_threshold: float = 0.8,
    resize_output: bool = True,
    verbose: bool = False,
) -> None:
    """Run file-based inference with a provided model and save results to disk."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    from utilities.inference import list_image_paths, save_diffuse_batch

    from ._shared import DEFAULT_IMAGE_EXTENSIONS, _resolve_device

    torch_device = getattr(model, "device", None)
    if torch_device is None:
        torch_device = torch.device(_resolve_device("cuda"))
    elif isinstance(torch_device, str):
        torch_device = torch.device(torch_device)

    target_side = getattr(model, "image_size", None) or getattr(
        model._model.dinov3.config, "image_size", 448
    )
    target_size = (target_side, target_side)

    input_dir = input_path if input_path.is_dir() else input_path.parent
    output_dir = output_path if output_path.is_dir() else output_path.parent
    image_paths = (
        [input_path]
        if input_path.is_file()
        else list_image_paths(input_dir, DEFAULT_IMAGE_EXTENSIONS)
    )

    model.eval()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        original_sizes = []
        for path in batch_paths:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                original_sizes.append(rgb_img.size[::-1])  # (H, W)
                tensor = TF.to_tensor(rgb_img)
                resized = TF.resize(tensor, target_size, antialias=True)
                batch_tensors.append(resized)

        rgb_batch = torch.stack(batch_tensors, dim=0).to(
            device=torch_device, dtype=torch.float32
        )
        inpaint_mask = (
            rgb_batch.mean(1, keepdim=True) > brightness_threshold
        ).to(device=torch_device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(
                images=rgb_batch,
                inpaint_mask_override=inpaint_mask,
                return_dict=True,
            )
        diffuse = outputs.get("diffuse")
        if diffuse is None:
            raise KeyError("Model output does not contain 'diffuse'")

        save_diffuse_batch(
            diffuse.clamp(0.0, 1.0),
            batch_paths,
            input_dir,
            output_dir,
            original_sizes=original_sizes if resize_output else None,
            resize_output=resize_output,
        )


def parse_cli():
    """Parse command line arguments and YAML file into inference options."""
    import yaml
    import argparse
    from inference import InferenceOptions

    parser = argparse.ArgumentParser(
        description="Run UnReflectAnything diffuse inference"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./config_inference.yaml",
        required=False,
        help="Absolute path to the inference YAML options file (default: ./config_inference.yaml)",
    )

    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading inference configuration from '{config_path}'")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_options = yaml.safe_load(handle)

    def _as_path(value: Optional[str]) -> Optional[Path]:
        return None if value is None else Path(value).expanduser().resolve()

    raw_weights = raw_options.get("weights_path")
    weights_path = _as_path(raw_weights)
    # Default to cache dir (same as download-weights) when omitted or set to "default"
    if weights_path is None or (
        isinstance(raw_weights, str) and str(raw_weights).strip().lower() == "default"
    ):
        try:
            from ._shared import (
                DEFAULT_WEIGHTS_FILENAME,
                get_cache_dir,
            )

            weights_path = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
        except ImportError:
            weights_path = None
    input_dir = _as_path(raw_options.get("input_dir"))
    output_dir = _as_path(raw_options.get("output_dir"))

    if weights_path is None or not weights_path.exists():
        cache_dir = None
        try:
            from ._shared import get_cache_dir

            cache_dir = get_cache_dir("weights")
        except ImportError:
            pass
        hint = (
            " Run 'unreflectanything download-weights' first, or set weights_path in the config."
            if cache_dir
            else ""
        )
        raise FileNotFoundError(
            f"weights_path must point to an existing checkpoint file.{hint}"
        )
    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError("input_dir must point to an existing directory")
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "✔️  Configuration loaded",
    )

    batch_size = int(raw_options.get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    image_extensions = raw_options.get("image_extensions")
    if image_extensions is None:
        extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        )
    else:
        extensions = tuple(ext.lower() for ext in image_extensions)

    resize_output = raw_options.get("resize_output", True)
    if not isinstance(resize_output, bool):
        raise ValueError("resize_output must be a boolean")

    brightness_threshold = float(raw_options.get("brightness_threshold", 0.7))
    if not (0.0 <= brightness_threshold <= 1.0):
        raise ValueError("brightness_threshold must be between 0.0 and 1.0")

    monitor_usage = raw_options.get("monitor_usage", False)
    if not isinstance(monitor_usage, bool):
        raise ValueError("monitor_usage must be a boolean")

    num_workers = int(raw_options.get("num_workers", 4))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    options = InferenceOptions(
        weights_path=weights_path,
        input_dir=input_dir,
        output_dir=output_dir,
        inpaint_mask_dilation=int(raw_options.get("inpaint_mask_dilation", 11)),
        run=raw_options.get("run"),
        runs_dir=_as_path(raw_options.get("runs_dir")),
        model_config_path=_as_path(raw_options.get("model_config_path")),
        model_module=raw_options.get("model_module"),
        batch_size=batch_size,
        device=raw_options.get("device", "cuda"),
        image_extensions=extensions,
        resize_output=resize_output,
        brightness_threshold=brightness_threshold,
        monitor_usage=monitor_usage,
        num_workers=num_workers,
    )
    return options
