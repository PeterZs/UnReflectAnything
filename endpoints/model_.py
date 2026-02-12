"""Model factory and UnReflectModel wrapper for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union, Any

from os import PathLike

if TYPE_CHECKING:
    from torch import Tensor


def _nn_module_base():
    import torch.nn as nn

    return nn.Module


def model(
    pretrained: bool = True,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    device: str = "cuda",
    config_path: Optional[Union[str, PathLike, Path, dict]] = None,
    verbose: bool = False,
    skip_path_resolution: bool = False,
    config: Optional[Union[dict, Any]] = None,
    weights: Optional[Union[str, PathLike, Path, dict, Any]] = None,
):
    """Return the model class or a pretrained model instance callable with batched RGB.

    Use this for a lightweight API: get a callable module and run it on tensors.

    Args:
        pretrained: If False , return the underlying model class
            (UnReflect_Model_TokenInpainter) for custom instantiation or training.
            If True (default), return an ``UnReflectModel`` instance with weights loaded,
            which you can call with a batched RGB tensor.
        weights_path: Path to checkpoint. Only used when pretrained=True.
            Ignored if ``weights`` is provided.
        weights: Optional weights: path to ``.pt``/``.pt`` file, or in-memory state dict.
            If provided, overrides ``weights_path``. Path operations apply only when it is a path.
        device: Device to load the model on when pretrained=True.
        config_path: Optional config source (YAML path) for architecture when
            loading from checkpoint. Only used when pretrained=True and ignored
            if ``config`` is provided.
        config: Optional in-memory configuration (dict or DotMap). If provided,
            this overrides ``config_path`` and is passed directly to the factory.
        verbose: If True, print progress information.

    Returns:
        If pretrained=False: the model class (UnReflect_Model_TokenInpainter).
        If pretrained=True: an ``UnReflectModel`` instance (nn.Module) that can
        be called with ``model(images)`` where images is [B, 3, H, W].
    """
    from utilities.config import (
        create_model_from_config,
        load_and_process_config,
        load_config_from_path_or_dict,
    )

    from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir, _resolve_device

    if config is not None:
        # Direct config (dict or DotMap) takes precedence; normalize to DotMap and set USE_TORCH_COMPILE=False
        model_config = load_config_from_path_or_dict(config)
        if model_config is None:
            raise ValueError("config could not be normalized (expected dict or DotMap).")
        if verbose:
            print("Using provided in-memory config (dict / DotMap).")
    else:
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
        import torch
        torch_device = torch.device(_resolve_device(device))
        return create_model_from_config(model_config, torch_device, verbose=verbose)

    # Weights: either path (str/Path) or in-memory state dict. Path ops only when path-like.
    chosen_weights = weights if weights is not None else weights_path

    def _is_path_like(x):
        return isinstance(x, (str, Path)) or getattr(x, "__fspath__", None) is not None

    weights_path_for_model = None
    state_dict_for_model = None
    if chosen_weights is not None:
        if _is_path_like(chosen_weights):
            weights_path_obj = Path(chosen_weights).expanduser()
            if not skip_path_resolution:
                resolved_weights = weights_path_obj.resolve()
            else:
                resolved_weights = weights_path_obj
            if not resolved_weights.exists():
                raise FileNotFoundError(
                    f"Weights file not found at '{resolved_weights}'.\n"
                    "Please run 'unreflect download --weights' or unreflectanything.download('weights') first."
                )
            name = weights_path_obj.name
            if (
                DEFAULT_WEIGHTS_FILENAME not in name
                and "full_model_weights" not in name
            ):
                raise ValueError(
                    f"Cannot find full model weights in '{weights_path_obj}'.\n"
                    "Please run 'unreflect download --weights' or unreflectanything.download('weights') first."
                )
            weights_path_for_model = resolved_weights
        else:
            state_dict_for_model = chosen_weights

    return UnReflectModel(
        pretrained=True,
        weights_path=weights_path_for_model,
        state_dict=state_dict_for_model,
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
        state_dict: Optional[Union[dict, Any]] = None,
    ):
        if not pretrained:
            raise ValueError(
                "UnReflectModel(pretrained=False) is not supported; use ura.model() to get the class."
            )
        super().__init__()
        import os
        from ._shared import (
            DEFAULT_WEIGHTS_FILENAME,
            get_cache_dir,
            _resolve_device,
        )

        torch_device = __import__("torch").device(_resolve_device(device))

        if state_dict is not None:
            from utilities.config import create_model_from_config

            inner = create_model_from_config(
                config_path, torch_device, verbose=verbose
            )
            sd = state_dict.get("model_state_dict", state_dict) if isinstance(state_dict, dict) else state_dict
            if isinstance(sd, dict) and sd:
                model_keys = set(inner.state_dict().keys())
                ckpt_keys = set(sd.keys())
                # Strip common wrapper prefixes when no key matches (e.g. module., model., _model.)
                for prefix in ("module.", "model.", "_model."):
                    if ckpt_keys and not (model_keys & ckpt_keys) and all(k.startswith(prefix) for k in ckpt_keys):
                        sd = {k.removeprefix(prefix): v for k, v in sd.items()}
                        if verbose:
                            print(f"State dict keys were prefixed with '{prefix}'; stripped to match model.")
                        break
            inner.load_state_dict(sd, strict=False)
            inner.eval()
        else:
            from utilities.model import load_pretrained

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

            inner = load_pretrained(
                weights_path=resolved_weights,
                config_path=config_path,
                device=str(torch_device),
                strict=False,
                verbose=verbose,
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
