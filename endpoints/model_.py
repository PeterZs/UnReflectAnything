"""Model factory and UnReflectModel wrapper for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from os import PathLike

if TYPE_CHECKING:
    from torch import Tensor


def _nn_module_base():
    import torch.nn as nn

    return nn.Module


def model(
    pretrained: bool = False,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    device: str = "cuda",
    config_path: Optional[Union[str, PathLike, Path, dict]] = None,
    verbose: bool = False,
):
    """Return the model class or a pretrained model instance callable with batched RGB.

    Use this for a lightweight API: get a callable module and run it on tensors.

    Args:
        pretrained: If False (default), return the underlying model class
            (UnReflect_Model_TokenInpainter) for custom instantiation or training.
            If True, return an ``UnReflectModel`` instance with weights loaded,
            which you can call with a batched RGB tensor.
        weights_path: Path to checkpoint. Only used when pretrained=True.
            Defaults to cache (~/.cache/unreflectanything/weights/full_model_weights.pt).
        device: Device to load the model on when pretrained=True.
        config_path: Optional config source (YAML path or dict) for architecture when
            loading from checkpoint. Only used when pretrained=True.
        verbose: If True, print progress information.

    Returns:
        If pretrained=False: the model class (UnReflect_Model_TokenInpainter).
        If pretrained=True: an ``UnReflectModel`` instance (nn.Module) that can
        be called with ``model(images)`` where images is [B, 3, H, W].
    """
    from utilities.config import create_model_from_config, load_and_process_config

    from unreflectanything._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir

    if config_path is None:
        config_path = (
            get_cache_dir("weights").parent / "configs" / "pretrained_config.yaml"
        )
    if (
        config_path is not None
        and hasattr(config_path, "is_dir")
        and getattr(config_path, "is_dir")()
    ):
        config_path = Path(config_path) / "pretrained_config.yaml"
    model_config = load_and_process_config(str(config_path))
    if verbose:
        print(f"Loaded model configuration from: `{config_path}`")

    if not pretrained:
        return create_model_from_config(model_config, device, verbose=verbose)

    if weights_path is not None:
        resolved_weights = Path(weights_path).expanduser().resolve()
        if not resolved_weights.exists():
            raise FileNotFoundError(
                f"Weights file not found at '{resolved_weights}'.\n"
                "Please run 'unreflect download --weights' or unreflectanything.download('weights') first."
            )
        if (
            DEFAULT_WEIGHTS_FILENAME not in resolved_weights.name
            and "full_model_weights" not in resolved_weights.name
        ):
            raise ValueError(
                f"Cannot find full model weights in '{resolved_weights}'.\n"
                "Please run 'unreflect download --weights' or unreflectanything.download('weights') first."
            )

    return UnReflectModel(
        pretrained=True,
        weights_path=weights_path,
        device=device,
        config_path=model_config,
        verbose=verbose,
    )


class UnReflectModel(_nn_module_base()):
    """Thin wrapper (nn.Module) around the loaded UnReflect model for tensor-in, tensor-out inference.

    Use ``ura.model(pretrained=True)`` to obtain an instance. The inner model is stored
    as a submodule so ``.to(device)``, ``.eval()``, and ``.parameters()`` work as expected.

    Attributes:
        image_size: Expected spatial size (side) for the inner encoder (e.g. 448).
        device: Device the model lives on (read-only).
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[Union[str, PathLike, Path]] = None,
        device: str = "cuda",
        config_path: Optional[Union[str, PathLike, Path, dict]] = None,
        verbose: bool = False,
    ):
        if not pretrained:
            raise ValueError(
                "UnReflectModel(pretrained=False) is not supported; use ura.model() to get the class."
            )
        super().__init__()
        from inference import load_pretrained
        import os
        from unreflectanything._shared import (
            DEFAULT_WEIGHTS_FILENAME,
            get_cache_dir,
            _resolve_device,
        )

        if weights_path is None:
            resolved_weights = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
        elif os.path.isdir(weights_path):
            resolved_weights = (
                Path(weights_path).expanduser().resolve() / DEFAULT_WEIGHTS_FILENAME
            )
        else:
            resolved_weights = Path(weights_path).expanduser().resolve()

        if not resolved_weights.exists():
            raise FileNotFoundError(
                f"Weights not found at {resolved_weights}. Run 'unreflect download --weights' first."
            )

        default_config_path = (
            get_cache_dir("weights").parent / "configs" / "pretrained_config.yaml"
        )
        torch_device = __import__("torch").device(_resolve_device(device))
        inner = load_pretrained(
            weights_path=resolved_weights,
            config_path=config_path,
            device=str(torch_device),
            strict=False,
            verbose=verbose,
            default_config_path=default_config_path
            if default_config_path.exists()
            else None,
        )
        self._model = inner
        self._device = torch_device
        cfg = self._model.dinov3.config
        self.image_size = getattr(
            cfg,
            "image_size",
            cfg.get("image_size", 448) if hasattr(cfg, "get") else 448,
        )

    @property
    def device(self):
        return self._device

    def forward(
        self,
        images: "Tensor",
        inpaint_mask_override: Optional["Tensor"] = None,
        return_dict: bool = False,
    ) -> Union["Tensor", Dict[str, "Tensor"]]:
        """Run inference on a batch of RGB images.

        Args:
            images: Batched RGB tensor [B, 3, H, W], values in [0, 1].
            inpaint_mask_override: Optional [B, 1, H, W] mask to force inpainting regions.
            return_dict: If True, return the full output dict; if False, return only diffuse tensor.

        Returns:
            If return_dict=False: diffuse tensor [B, 3, H, W].
            If return_dict=True: dict with at least ``diffuse``, ``highlight``, etc.
        """
        import torch

        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must be [B, 3, H, W], got shape {tuple(images.shape)}"
            )
        batch = {
            "rgb": images.to(device=self._device, dtype=torch.float32),
        }
        if inpaint_mask_override is not None:
            batch["inpaint_mask_override"] = inpaint_mask_override.to(
                device=self._device, dtype=torch.float32
            )
        self._model.eval()
        with torch.no_grad():
            out = self._model(batch)
        diffuse = out.get("diffuse")
        highlight = out.get("highlight")
        if diffuse is None:
            raise KeyError("Model output does not contain 'diffuse'")
        diffuse = diffuse.clamp(0.0, 1.0)
        if return_dict:
            out["diffuse"] = diffuse
            out["highlight"] = highlight
            return out
        return diffuse

    def eval(self):
        """Set the inner model to eval mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True):
        """Set the inner model to train mode (for fine-tuning)."""
        self._model.train(mode)
        return self
