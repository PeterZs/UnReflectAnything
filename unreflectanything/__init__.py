"""UnReflectAnything: deep learning method for removing specular reflections from RGB images.

Public API:
    run_pipeline: Train or test the network (main entry for training/testing).
    run_inference: Run inference on an image directory from InferenceOptions.
    InferenceOptions: Dataclass for inference configuration.
    compute_highlight_mask: Compute binary highlight masks from RGB batch.
    get_weights_cache_dir: Default cache directory for downloaded weights.
"""

from __future__ import annotations

__all__ = [
    "run_pipeline",
    "run_inference",
    "InferenceOptions",
    "compute_highlight_mask",
    "get_weights_cache_dir",
]


def __getattr__(name: str):
    """Lazy imports so that `import unreflectanything` and CLI --help stay fast."""
    if name == "run_pipeline":
        import main as _main
        return _main.run_pipeline
    if name == "InferenceOptions":
        from inference import InferenceOptions
        return InferenceOptions
    if name == "compute_highlight_mask":
        from inference import compute_highlight_mask
        return compute_highlight_mask
    if name == "run_inference":
        from inference import run_inference
        return run_inference
    if name == "get_weights_cache_dir":
        from unreflectanything.weights import get_weights_cache_dir
        return get_weights_cache_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
