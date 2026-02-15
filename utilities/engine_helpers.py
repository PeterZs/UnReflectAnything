"""
Helpers for the training Engine: compositing, metrics formatting, image conversion,
visualization logging, tensor cleanup, and checkpoint loading.
"""
import os
from typing import Any, Callable, Optional

import pandas as pd
import torch
import torchvision.transforms as transforms
import wandb


def composite_specular_diffuse(
    specular: torch.Tensor, diffuse: torch.Tensor
) -> torch.Tensor:
    """
    Composite specular and diffuse components into a reconstructed image.

    Args:
        specular: [B, C, H, W] or [B, 4, H, W] for RGBA
        diffuse: [B, C, H, W] or [B, 4, H, W] for RGBA

    Returns:
        Reconstructed RGB [B, 3, H, W]
    """
    if specular.shape[1] == 4 and diffuse.shape[1] == 4:
        spec_rgb = specular[:, :3]
        spec_alpha = specular[:, 3:4]
        diff_rgb = diffuse[:, :3]
        diff_alpha = diffuse[:, 3:4]
        recon_rgb = spec_rgb * spec_alpha + diff_rgb * diff_alpha * (1 - spec_alpha)
        recon_rgb = torch.clamp(recon_rgb, 0, 1)
        return recon_rgb
    recon = specular + diffuse
    recon = recon / recon.max()
    return torch.clamp(recon, 0, 1)


def prepare_metrics_for_wandb(metrics: dict, phase: str) -> dict:
    """Format metrics dict with phase prefix for wandb logging."""
    formatted = {}
    for k, v in metrics.items():
        if "Step" in k:
            formatted[k] = v
        else:
            formatted[f"{phase}/{k}"] = v
    return formatted


def to_cpu_image(tensor: Optional[torch.Tensor], batch_idx: int = 0):
    """Convert tensor to PIL Image; handles [B,C,H,W], [C,H,W], [H,W]."""
    if tensor is None:
        return None
    if tensor.dim() == 4:
        tensor = tensor[0].clone()
    elif tensor.dim() == 3:
        tensor = tensor.clone()
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).clone()
    tensor = tensor.cpu().detach().clamp(0, 1)
    pil_image = transforms.ToPILImage()(tensor)
    del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pil_image


def add_image_safely(
    viz_dict: dict,
    key: str,
    tensor: Optional[torch.Tensor],
    caption: str,
    batch_idx: int = 0,
    phase: Optional[str] = None,
    test_idx: Optional[int] = None,
    metrics_test_df: Optional[pd.DataFrame] = None,
) -> None:
    """Add image to viz dict and log to wandb with phase/test_idx prefixing."""
    if tensor is None:
        return
    image = wandb.Image(to_cpu_image(tensor, batch_idx))
    log_key = key
    if isinstance(phase, str):
        if phase == "Test" and test_idx is not None:
            if key.startswith("Test/") and not key.startswith(f"Test/test_idx_{test_idx}/"):
                log_key = key.replace("Test/", f"Test/test_idx_{test_idx}/", 1)
            elif not key.startswith(f"Test/test_idx_{test_idx}/"):
                log_key = f"Test/test_idx_{test_idx}/{key}"
        elif not key.startswith(("Training/", "Validation/", "Test/")):
            log_key = f"{phase}/{key}"
    viz_dict[log_key] = image
    payload = {log_key: image}
    if phase == "Test" and test_idx is not None:
        payload["Step/test_idx"] = int(test_idx)
    elif (
        metrics_test_df is not None
        and not metrics_test_df.empty
        and "Step/test_idx" in metrics_test_df.columns
    ):
        try:
            payload["Step/test_idx"] = int(metrics_test_df["Step/test_idx"].iloc[-1])
        except Exception:
            pass
    wandb.log(payload)


def cleanup_tensor_dict(
    tensor_dict: Optional[dict], keys_to_keep: Optional[list] = None
) -> None:
    """In-place cleanup of tensor dict; optionally keep specified keys."""
    if tensor_dict is None:
        return
    keys_to_keep = keys_to_keep or []
    to_delete = [k for k in tensor_dict.keys() if k not in keys_to_keep]
    for key in to_delete:
        if isinstance(tensor_dict[key], torch.Tensor):
            tensor_dict[key].detach_()
            if tensor_dict[key].is_cuda:
                tensor_dict[key].cpu()
            del tensor_dict[key]
        elif isinstance(tensor_dict[key], dict):
            cleanup_tensor_dict(tensor_dict[key])
            del tensor_dict[key]


def create_model_from_checkpoint(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> Optional[dict]:
    """
    Load checkpoint and return model + state dicts. Does not build model class;
    returns dict with model_state_dict, optimizer_state_dict, config, epoch, etc.
    """
    if not os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".pt"):
            alt = checkpoint_path[:-3] + ".pth"
        elif checkpoint_path.endswith(".pth"):
            alt = checkpoint_path[:-4] + ".pt"
        else:
            return None
        if os.path.exists(alt):
            checkpoint_path = alt
        else:
            return None

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_class_name = checkpoint.get("model_class_name")
    model_class_module = checkpoint.get("model_class_module")

    if not (model_class_name and model_class_module):
        return None

    import importlib
    module = importlib.import_module(model_class_module)
    model_class = getattr(module, model_class_name)
    model = model_class()
    model.load_state_dict(checkpoint["model_state_dict"])
    if device:
        model = model.to(device)

    return {
        "model": model,
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        "LRscheduler_state_dict": checkpoint.get("LRscheduler_state_dict"),
        "LRschedulerPlateau_state_dict": checkpoint.get("LRschedulerPlateau_state_dict"),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config"),
        "model_class_name": model_class_name,
        "model_class_module": model_class_module,
    }
