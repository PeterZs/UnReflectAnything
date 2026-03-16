"""Model factory and UnReflectModel wrapper for UnReflectAnything.

Provides two public symbols:

* :func:`model` – thin convenience factory (delegates to :class:`UnReflectModel`).
* :class:`UnReflectModel` – ``nn.Module`` wrapper for tensor-in / tensor-out
  inference **and** weight-free instantiation for training.

Resolution order (applies to both config and weights):

1. In-memory object (``config`` / ``weights``)  – used directly.
2. Filesystem path   (``config_path`` / ``weights_path``) – loaded from disk.
3. Default cached path in ``~/.cache/unreflectanything/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union, Any

from os import PathLike

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nn_module_base():
    """Lazy-import ``nn.Module`` so ``torch`` is not loaded at module level."""
    import torch.nn as nn

    return nn.Module


def _is_path(x) -> bool:
    """Return True when *x* looks like a filesystem path (str / Path / os.PathLike)."""
    return isinstance(x, (str, Path)) or getattr(x, "__fspath__", None) is not None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def model(
    pretrained: bool = True,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    device: str = "cuda",
    config_path: Optional[Union[str, PathLike, Path]] = None,
    verbose: bool = False,
    config: Optional[Union[dict, Any]] = None,
    weights: Optional[Union[str, PathLike, Path, dict, Any]] = None,
    strict: bool = True,
) -> "UnReflectModel":
    """Convenience factory — delegates entirely to :class:`UnReflectModel`.

    See :meth:`UnReflectModel.__init__` for full parameter documentation.

    Returns:
        An ``UnReflectModel`` instance (with or without pretrained weights).
    """
    return UnReflectModel(
        pretrained=pretrained,
        weights_path=weights_path,
        device=device,
        config_path=config_path,
        verbose=verbose,
        config=config,
        weights=weights,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class UnReflectModel(_nn_module_base()):
    """``nn.Module`` wrapper around the inner UnReflect model.

    The inner model is stored as the ``_model`` submodule so that ``.to()``,
    ``.eval()``, ``.parameters()``, and ``state_dict()`` propagate correctly.

    Attributes:
        image_size (int): Spatial size expected by the encoder (e.g. 448).
        device (torch.device): Device the model lives on (read-only property).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[Union[str, PathLike, Path]] = None,
        device: str = "cuda",
        config_path: Optional[Union[str, PathLike, Path]] = None,
        verbose: bool = False,
        config: Optional[Union[dict, Any]] = None,
        weights: Optional[Union[str, PathLike, Path, dict, Any]] = None,
        strict: bool = True,
    ):
        """Build (and optionally load weights for) the UnReflect model.

        Args:
            pretrained: When ``True`` (default) load checkpoint weights after
                building the architecture.  When ``False`` only build the
                architecture — useful for custom training or fine-tuning.
            weights_path: Path to a ``.pt`` checkpoint file.  Falls back to the
                cached ``full_model_weights.pt`` when ``None``.
                *Ignored when* ``weights`` *is provided.*
            device: Target device string (``"cuda"``, ``"cpu"``, ``"cuda:1"``, …).
            config_path: Path to a YAML config describing the architecture.
                Falls back to the cached ``pretrained_config.yaml`` when ``None``.
                *Ignored when* ``config`` *is provided.*
            verbose: Print progress messages.
            config: In-memory configuration (dict or DotMap).
                **Overrides** ``config_path`` when provided.
            weights: In-memory state-dict **or** path to a checkpoint.
                **Overrides** ``weights_path`` when provided.
        """
        super().__init__()
        import torch
        import importlib
        from ._shared import _resolve_device
        from ._model_builder import load_config_minimal, create_model_from_config_minimal
        # ── 1. Resolve target device ──────────────────────────────────
        torch_device = torch.device(_resolve_device(device))

        # ── 2. Resolve & load model configuration ─────────────────────
        #    `config` (object) takes priority over `config_path` (file).
        if config is None and config_path is None:
            pkg = importlib.resources.files("unreflectanything")
            config_path = pkg / "assets" / "pretrained_config.yaml"
        model_config = self._resolve_config(config, config_path, verbose)
        # ── 3. Build model architecture from config (no utilities/engine imports) ─
        inner = create_model_from_config_minimal(model_config, torch_device, verbose=verbose)
        # ── 4. Load pretrained weights (skipped when pretrained=False) ─
        if pretrained:
            sd = self._resolve_and_load_weights(
                weights, weights_path, torch_device, verbose,
            )
            sd = self._strip_key_prefixes(inner, sd, verbose)
            inner.load_state_dict(sd, strict=strict)
            inner.eval()
            if verbose:
                print("Pretrained weights loaded; model set to eval mode.")

        # ── 5. Register submodule & cache metadata ────────────────────
        self._model = inner
        self._device = torch_device

        # Read the spatial size the encoder expects (e.g. 448)
        encoder_cfg = self._model.dinov3.config
        self.image_size = getattr(
            encoder_cfg,
            "image_size",
            encoder_cfg.get("image_size", 448)
            if hasattr(encoder_cfg, "get")
            else 448,
        )

    # ------------------------------------------------------------------
    # Internal resolution helpers (static — no instance state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(config, config_path, verbose):
        """Return a parsed DotMap config (minimal loader; no utilities import)."""

        from ._shared import DEFAULT_CONFIG_FILENAME, get_cache_dir
        from ._model_builder import load_config_minimal

        # --- In-memory config takes priority ---
        if config is not None:
            parsed = load_config_minimal(config=config)
            if parsed is None:
                raise ValueError(
                    "config could not be parsed (expected dict or DotMap)."
                )
            if verbose:
                print("Loaded model configuration from in-memory object.")
            return parsed

        # --- Fall back to filesystem path ---
        if config_path is None:
            config_path = get_cache_dir("configs") / DEFAULT_CONFIG_FILENAME
        config_path = Path(config_path)
        if config_path.is_dir():
            config_path = config_path / DEFAULT_CONFIG_FILENAME

        parsed = load_config_minimal(config_path=config_path)
        if verbose:
            print(f"Loaded model configuration from: `{config_path}`")
        return parsed

    @staticmethod
    def _resolve_and_load_weights(weights, weights_path, device, verbose):
        """Return a flat ``state_dict`` ready for ``load_state_dict``.

        Resolution order:
            1. ``weights`` is a dict (state-dict) → returned as-is.
            2. ``weights`` is a path               → loaded from disk.
            3. ``weights_path`` (explicit path)     → loaded from disk.
            4. Default cached weights file.

        Automatically unwraps training-checkpoint wrappers
        (e.g. ``{"model_state_dict": {...}, ...}``).
        """
        import torch
        from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir

        # Pick the authoritative source: `weights` overrides `weights_path`
        source = weights if weights is not None else weights_path

        # Default to the cached checkpoint when nothing was provided
        if source is None:
            source = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME

        # ── In-memory state-dict ──
        if not _is_path(source):
            sd = source
            if isinstance(sd, dict):
                sd = sd.get("model_state_dict", sd)
            if verbose:
                print("Using in-memory state dict.")
            return sd

        # ── Filesystem path ──
        resolved = Path(source).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Weights file not found at '{resolved}'.\n"
                "Run 'unreflectanything download --weights' or "
                "unreflectanything.download('weights') first."
            )
        if verbose:
            print(f"Loading weights from: `{resolved}`")
        
        checkpoint = torch.load(resolved, map_location=device, weights_only=False)

        # Unwrap training-checkpoint wrapper if present
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("model_state_dict", checkpoint)
        return checkpoint

    @staticmethod
    def _strip_key_prefixes(model_module, state_dict, verbose):
        """Strip DDP / wrapper prefixes from checkpoint keys when they don't
        match the model's own parameter names.

        Handles common prefixes: ``module.``, ``model.``, ``_model.``.
        """
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict

        model_keys = set(model_module.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # If at least some keys already match, no stripping needed
        if model_keys & ckpt_keys:
            return state_dict

        for prefix in ("module.", "model.", "_model."):
            if all(k.startswith(prefix) for k in ckpt_keys):
                state_dict = {
                    k.removeprefix(prefix): v for k, v in state_dict.items()
                }
                if verbose:
                    print(f"Stripped '{prefix}' prefix from state-dict keys.")
                break
        return state_dict

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self):
        """``torch.device`` the model lives on."""
        return self._device

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: "Tensor",
        threshold: float = 0.3,
        dilation = 40,
        inpaint_mask_override: Optional["Tensor"] = None,
        return_dict: bool = False,
    ) -> Union["Tensor", Dict[str, "Tensor"]]:
        """Run inference on a batch of RGB images.

        Args:
            images: ``[B, 3, H, W]`` float tensor with values in ``[0, 1]``.
            inpaint_mask_override: Optional ``[B, 1, H, W]`` binary mask to
                force specific inpainting regions.
            return_dict: If ``True`` return the full output dict; otherwise
                return only the diffuse tensor.

        Returns:
            ``[B, 3, H, W]`` diffuse tensor (clamped to [0, 1]), or the full
            output dict when ``return_dict=True``.
        """
        import torch

        # Validate input shape
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must be [B, 3, H, W], got shape {tuple(images.shape)}"
            )

        # Build the batch dict expected by the inner model
        batch = {
            "rgb": images.to(device=self._device, dtype=torch.float32),  # [B,3,H,W]
            "inpaint_mask_threshold": threshold,
            "inpaint_mask_dilation": dilation,
        }
        if inpaint_mask_override is not None:
            batch["inpaint_mask_override"] = inpaint_mask_override.to(
                device=self._device, dtype=torch.float32,
            )  # [B,1,H,W]

        # Inference (always in eval / no-grad)
        self._model.eval()
        with torch.no_grad():
            out = self._model(batch)

        # Extract and clamp the diffuse component
        diffuse = out.get("diffuse")  # [B,3,H,W]
        highlight = out.get("highlight")  # [B,1,H,W]
        diffuse = diffuse.clamp(0.0, 1.0)  # [B,3,H,W]

        if return_dict:
            out["diffuse"] = diffuse
            out["highlight"] = highlight
            return out
        return diffuse

    # ------------------------------------------------------------------
    # Train / eval mode helpers
    # ------------------------------------------------------------------

    def eval(self):
        """Set the inner model to eval mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True):
        """Set the inner model to train mode (for fine-tuning)."""
        self._model.train(mode)
        return self
