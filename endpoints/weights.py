"""Download and resolve pretrained weights and assets for UnReflectAnything.

This module provides functions to download pretrained model weights, sample images,
and example notebooks from the HuggingFace Hub.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Default Hugging Face repo for assets (override with UNREFLECTANYTHING_WEIGHTS_REPO)
DEFAULT_WEIGHTS_REPO = "AlbeRota/UnReflectAnything"

# Default checkpoint filename under the cache dir (must match what the HF repo provides)
DEFAULT_WEIGHTS_FILENAME = "weights_best.pt"

# Subfolder names in the HuggingFace repo
IMAGES_SUBFOLDER = "sample_images"
NOTEBOOKS_SUBFOLDER = "notebooks"


def get_cache_dir() -> Path:
    """Return the default base directory for caching downloaded assets (cross-platform).

    This is the parent directory for weights, images, and notebooks.

    - **Linux / macOS**: Uses ``$XDG_CACHE_HOME`` if set (XDG Base Dir spec), otherwise
      ``~/.cache``. Result: ``$XDG_CACHE_HOME/unreflectanything`` or
      ``~/.cache/unreflectanything``.
    - **Windows**: Uses ``%LOCALAPPDATA%`` if set (e.g. ``C:\\Users\\...\\AppData\\Local``),
      otherwise ``~/.cache`` (``~`` expands to the user profile). Result:
      ``%LOCALAPPDATA%\\unreflectanything`` or ``~/.cache/unreflectanything``.

    Returns:
        Path to the base cache directory.
    """
    if os.name == "nt":
        # Windows: use AppData\Local (standard cache location)
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.cache"))
    else:
        # Linux, macOS, etc.: XDG_CACHE_HOME or ~/.cache
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base).expanduser().resolve() / "unreflectanything"


def get_weights_cache_dir() -> Path:
    """Return the default directory for caching downloaded weights (cross-platform).

    - **Linux / macOS**: Uses ``$XDG_CACHE_HOME`` if set (XDG Base Dir spec), otherwise
      ``~/.cache``. Result: ``$XDG_CACHE_HOME/unreflectanything/weights`` or
      ``~/.cache/unreflectanything/weights``.
    - **Windows**: Uses ``%LOCALAPPDATA%`` if set (e.g. ``C:\\Users\\...\\AppData\\Local``),
      otherwise ``~/.cache`` (``~`` expands to the user profile). Result:
      ``%LOCALAPPDATA%\\unreflectanything\\weights`` or ``~/.cache/unreflectanything/weights``.

    Returns:
        Path to the weights cache directory.
    """
    return get_cache_dir() / "weights"


def get_images_cache_dir() -> Path:
    """Return the default directory for caching downloaded sample images.

    Returns:
        Path to the images cache directory.
    """
    return get_cache_dir() / "images"


def get_notebooks_cache_dir() -> Path:
    """Return the default directory for caching downloaded notebooks.

    Returns:
        Path to the notebooks cache directory.
    """
    return get_cache_dir() / "notebooks"


def _ensure_huggingface_hub():
    """Ensure huggingface_hub is installed, exit with helpful message if not."""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        return snapshot_download, hf_hub_download
    except ImportError:
        print(
            "The 'huggingface_hub' package is required for downloading assets. "
            "Install it with: pip install unreflectanything[weights]",
            file=sys.stderr,
        )
        sys.exit(1)


def download_weights(
    output_dir: Optional[Path] = None,
    variant: str = "default",
    force: bool = False,
) -> Path:
    """Download pretrained weights to the given directory.

    Weights are hosted on the Hugging Face Hub. Requires the optional
    dependency: ``pip install unreflectanything[weights]``.

    Args:
        output_dir: Directory to save weights. If None, uses the default cache
            directory (``~/.cache/unreflectanything/weights``).
        variant: Weights variant to download (e.g. ``default`` for main checkpoint).
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory where weights were saved.
    """
    snapshot_download, _ = _ensure_huggingface_hub()

    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    
    if output_dir is None:
        output_dir = get_weights_cache_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    revision = None if variant == "default" else variant
    
    # Download only the weights file, not the entire repo
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        revision=revision,
        force_download=force,
        allow_patterns=["*.pt", "*.pth", "*.bin", "*.safetensors", "config*.yaml", "*.json"],
        ignore_patterns=[IMAGES_SUBFOLDER + "/*", NOTEBOOKS_SUBFOLDER + "/*"],
    )
    print(f"Weights saved to {output_dir}")
    return output_dir


def download_images(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download sample images for testing the model.

    Sample images are hosted in a subfolder of the HuggingFace repo.

    Args:
        output_dir: Directory to save images. If None, uses the default cache
            directory (``~/.cache/unreflectanything/images``).
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory where images were saved.
    """
    snapshot_download, _ = _ensure_huggingface_hub()

    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    
    if output_dir is None:
        output_dir = get_images_cache_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download only the sample_images subfolder
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        force_download=force,
        allow_patterns=[f"{IMAGES_SUBFOLDER}/*"],
    )
    
    # Check if files were downloaded to a subfolder and move them up
    subfolder_path = output_dir / IMAGES_SUBFOLDER
    if subfolder_path.exists() and subfolder_path.is_dir():
        # Move files from subfolder to output_dir
        for item in subfolder_path.iterdir():
            target = output_dir / item.name
            if not target.exists() or force:
                item.rename(target)
        # Remove empty subfolder
        try:
            subfolder_path.rmdir()
        except OSError:
            pass  # Not empty or other issue, leave it
    
    print(f"Sample images saved to {output_dir}")
    return output_dir


def download_notebooks(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download example Jupyter notebooks.

    Notebooks are hosted in a subfolder of the HuggingFace repo.

    Args:
        output_dir: Directory to save notebooks. If None, uses the default cache
            directory (``~/.cache/unreflectanything/notebooks``).
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory where notebooks were saved.
    """
    snapshot_download, _ = _ensure_huggingface_hub()

    repo_id = os.environ.get("UNREFLECTANYTHING_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    
    if output_dir is None:
        output_dir = get_notebooks_cache_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download only the notebooks subfolder
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        force_download=force,
        allow_patterns=[f"{NOTEBOOKS_SUBFOLDER}/*", f"{NOTEBOOKS_SUBFOLDER}/**/*"],
    )
    
    # Check if files were downloaded to a subfolder and move them up
    subfolder_path = output_dir / NOTEBOOKS_SUBFOLDER
    if subfolder_path.exists() and subfolder_path.is_dir():
        # Move files from subfolder to output_dir
        for item in subfolder_path.iterdir():
            target = output_dir / item.name
            if not target.exists() or force:
                if item.is_file():
                    item.rename(target)
                elif item.is_dir():
                    import shutil
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(item), str(target))
        # Remove empty subfolder
        try:
            subfolder_path.rmdir()
        except OSError:
            pass  # Not empty or other issue, leave it
    
    print(f"Notebooks saved to {output_dir}")
    return output_dir


def download_all(
    output_dir: Optional[Path] = None,
    variant: str = "default",
    force: bool = False,
) -> Path:
    """Download all assets: weights, sample images, and notebooks.

    Args:
        output_dir: Base directory to save all assets. If None, uses the default
            cache directory (``~/.cache/unreflectanything/``).
        variant: Weights variant to download.
        force: If True, re-download even if files already exist.

    Returns:
        Path to the base directory where assets were saved.
    """
    if output_dir is None:
        output_dir = get_cache_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_weights(output_dir=output_dir / "weights", variant=variant, force=force)
    download_images(output_dir=output_dir / "images", force=force)
    download_notebooks(output_dir=output_dir / "notebooks", force=force)
    
    print(f"All assets saved to {output_dir}")
    return output_dir
