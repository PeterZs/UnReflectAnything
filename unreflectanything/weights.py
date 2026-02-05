"""Download and resolve pretrained weights for UnReflectAnything."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default Hugging Face repo for weights (override with UNREFLECTANYTHING_WEIGHTS_REPO)
DEFAULT_WEIGHTS_REPO = "AlbeRota/UnReflectAnything"


def get_weights_cache_dir() -> Path:
    """Return the default directory for caching downloaded weights.

    Uses ``$XDG_CACHE_HOME/unreflectanything/weights`` if set, otherwise
    ``~/.cache/unreflectanything/weights``.
    """
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base).expanduser().resolve() / "unreflectanything" / "weights"


def download_weights(
    output_dir: Path,
    variant: str = "default",
    force: bool = False,
) -> None:
    """Download pretrained weights to the given directory.

    Weights are hosted on the Hugging Face Hub. Requires the optional
    dependency: ``pip install unreflectanything[weights]``.

    Args:
        output_dir: Directory to save weights (e.g. from ``get_weights_cache_dir()``).
        variant: Weights variant to download (e.g. ``default`` for main checkpoint).
        force: If True, re-download even if files already exist.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "The 'huggingface_hub' package is required for download-weights. "
            "Install it with: pip install unreflectanything[weights]",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    revision = None if variant == "default" else variant
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        revision=revision,
        force_download=force,
    )
    print(f"Weights saved to {output_dir}")
