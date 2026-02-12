"""Verification API (dataset structure and weights) for UnReflectAnything."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from os import PathLike


def _verify_weights_impl(
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    model_config_path: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify weights file exists and loads into model with no key alignment errors."""
    import torch
    from utilities.model import load_pretrained

    from ._shared import DEFAULT_WEIGHTS_FILENAME, get_cache_dir

    if weights_path is None:
        resolved = get_cache_dir("weights") / DEFAULT_WEIGHTS_FILENAME
    else:
        resolved = Path(weights_path).expanduser().resolve()

    if not resolved.exists():
        print(f"Weights file not found: {resolved}")
        return False
    print(
        f"Found weights file: {resolved}\nLoading weights and verifying key alignments..."
    )

    config_path = Path(model_config_path).expanduser().resolve() if model_config_path else None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_pretrained(
            weights_path=resolved,
            config_path=config_path,
            device=str(device),
            strict=True,
            verbose=False,
        )
        print("✔️  Weights verified: loaded into model with no key alignment errors.")
        return True
    except (KeyError, RuntimeError, FileNotFoundError) as e:
        print(f"❌  Weights verification failed: {e}")
        print("Download the model weights with 'unreflect download --weights'")
        return False


def _verify_dataset_impl(
    dataset_path: Path,
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Internal implementation of dataset verification."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    from dataset import (
        HOUSECAT6D_Dataset,
        POLARGB_Dataset,
        UnReflectAnything_Dataset,
        SCRREAM_Dataset,
    )

    dataset_classes = {
        "SCRREAM": SCRREAM_Dataset,
        "HOUSECAT6D": HOUSECAT6D_Dataset,
        "POLARGB": POLARGB_Dataset,
        "RGBP": UnReflectAnything_Dataset,
    }

    if dataset_type is None:
        for name, cls in dataset_classes.items():
            try:
                ds = cls(
                    root_dir=str(dataset_path),
                    target_size=(224, 224),
                    few_images=True,
                )
                if len(ds) > 0:
                    print(f"Detected dataset type: {name}")
                    print(f"Found {len(ds)} samples")
                    return True
            except Exception:
                continue
        print("Could not auto-detect dataset type")
        return False

    dataset_type_upper = dataset_type.upper()
    if dataset_type_upper not in dataset_classes:
        print(f"Unknown dataset type: {dataset_type}")
        print(f"Available types: {list(dataset_classes.keys())}")
        return False

    cls = dataset_classes[dataset_type_upper]
    try:
        ds = cls(
            root_dir=str(dataset_path),
            target_size=(224, 224),
            few_images=True,
        )
        sample_count = len(ds)
        if sample_count > 0:
            print(f"Dataset '{dataset_type}' verified successfully!")
            print(f"Found {sample_count} samples")
            try:
                _ = ds[0]
                print("Sample loading: OK")
            except Exception as e:
                print(f"Warning: Sample loading failed: {e}")
                return False
            return True
        else:
            print(f"Dataset '{dataset_type}' has no samples")
            return False
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        return False


def verify(
    what: Literal["dataset", "weights"],
    path: Optional[Union[str, PathLike, Path]] = None,
    weights_path: Optional[Union[str, PathLike, Path]] = None,
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
    model_config_path: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify either dataset structure or weights integrity.

    - **dataset**: Checks that the directory at `path` has the expected
      structure for the given dataset type (or auto-detects). Requires `path`.
    - **weights**: Checks that the weights file exists and loads into the
      model with no state_dict key alignment errors.

    Args:
        what: Either "dataset" or "weights".
        path: Dataset root directory (required when what="dataset").
        weights_path: Path to weights file (optional when what="weights").
        dataset_type: Dataset type for dataset verification.
        config: Optional config for dataset verification.
        model_config_path: Optional model config YAML for weights verification.

    Returns:
        True if verification passed, False otherwise.
    """
    if what == "dataset":
        if path is None:
            raise ValueError("path is required when what='dataset'")
        return _verify_dataset_impl(
            path=Path(path).expanduser().resolve(),
            dataset_type=dataset_type,
            config=config,
        )
    elif what == "weights":
        return _verify_weights_impl(
            weights_path=weights_path,
            model_config_path=model_config_path,
        )
    else:
        raise ValueError(f"what must be 'dataset' or 'weights', got {what!r}")


def verify_dataset(
    path: Union[str, PathLike, Path],
    dataset_type: Optional[str] = None,
    config: Optional[Union[str, PathLike, Path]] = None,
) -> bool:
    """Verify that a dataset has the correct structure for training/testing.

    Convenience wrapper around verify(what="dataset", path=path, ...).
    """
    return verify(
        what="dataset",
        path=path,
        dataset_type=dataset_type,
        config=config,
    )
