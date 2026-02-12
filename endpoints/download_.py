"""Download assets API for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from os import PathLike

from ._shared import (
    download_configs,
    download_images,
    download_notebooks,
    download_weights,
    get_cache_dir,
)


def download(
    what: Optional[Literal["weights", "images", "notebooks", "configs", "all"]] = None,
    *,
    asset: Optional[Literal["weights", "images", "notebooks", "configs", "all"]] = None,
    output_dir: Optional[Union[str, PathLike, Path]] = None,
    variant: str = "default",
    force: bool = False,
) -> Path:
    """Download assets from the HuggingFace repository.

    Same behavior as the CLI ``unreflect download --weights`` / ``--images`` /
    ``--notebooks`` / ``--all``.

    Args:
        what: What to download (positional). Use ``asset=`` as alias.
        asset: Alias for ``what``.
        output_dir: Directory to save downloaded files. If None, uses the
            default cache directory (~/.cache/unreflectanything/).
        variant: Weights variant to download (e.g., "default").
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory where files were saved.
    """
    resolved_what = asset if asset is not None else what
    if resolved_what is None:
        resolved_what = "weights"
    if resolved_what not in ("weights", "images", "notebooks", "configs", "all"):
        raise ValueError(
            f"Invalid 'what'/'asset' value: {resolved_what!r}. "
            "Must be 'weights', 'images', 'notebooks', 'configs', or 'all'."
        )

    if output_dir is None:
        output_path = get_cache_dir("weights").parent
    else:
        output_path = Path(output_dir).expanduser().resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    if resolved_what == "weights":
        weights_dir = output_path / "weights"
        download_weights(output_dir=weights_dir, variant=variant, force=force)
        return weights_dir
    elif resolved_what == "images":
        images_dir = output_path / "images"
        download_images(output_dir=images_dir, force=force)
        return images_dir
    elif resolved_what == "notebooks":
        notebooks_dir = output_path / "notebooks"
        download_notebooks(output_dir=notebooks_dir, force=force)
        return notebooks_dir
    elif resolved_what == "configs":
        configs_dir = output_path / "configs"
        download_configs(output_dir=configs_dir, force=force)
        return configs_dir
    else:  # "all"
        download_weights(
            output_dir=output_path / "weights", variant=variant, force=force
        )
        download_images(output_dir=output_path / "images", force=force)
        download_notebooks(output_dir=output_path / "notebooks", force=force)
        download_configs(output_dir=output_path / "configs", force=force)
        return output_path
