from typing import Dict, Any, Optional, List
import torch
import yaml
from dotmap import DotMap
from torch.utils.data import ConcatDataset

from utilities.model import DataParallelWrapper
import ast
import importlib
import time
from logger import get_logger
import json

logger = get_logger(__name__).set_context("CONFIG")

# Allowed values for DISTRIBUTE (training distribution mode)
DISTRIBUTE_SINGLEGPU = "single"
DISTRIBUTE_DP = "dp"
DISTRIBUTE_DDP = "ddp"


def resolve_distribute(config: DotMap) -> None:
    """
    Resolve and normalize config.DISTRIBUTE from config.
    Mutates config so that config.get("DISTRIBUTE", "single") is always one of
    "single", "dp", "ddp".
    """
    distribute_entry = config.get("DISTRIBUTE", "single")
    if distribute_entry == "single" and torch.cuda.device_count() > 1:
        logger.warning(
            f"DISTRIBUTE is 'singlegpu' even though multiple GPUs are available"
        )

    if distribute_entry is not None:
        normalized = str(distribute_entry).strip().lower()
        if normalized not in (DISTRIBUTE_SINGLEGPU, DISTRIBUTE_DP, DISTRIBUTE_DDP):
            raise ValueError(
                f"DISTRIBUTE must be one of 'singlegpu', 'DP', 'DDP'; got {distribute_entry!r}"
            )
        config["DISTRIBUTE"] = normalized
        
    return config

def create_model_from_config(
    config: DotMap,
    device: torch.device,
    verbose: bool = True,
):
    """
    Create the RGBPOLDecomposer model from configuration.
    
    This function initializes a RGBPOLDecomposer model by extracting configuration
    parameters for different components (RGB encoder, POL encoder, cross-attention,
    and decoders) and creates the model with the specified architecture.

    Args:
        config (DotMap): Configuration dictionary containing model parameters including:
            - MODEL: Model architecture configuration
            - DATASETS: Dataset configurations for extracting target image size
            - USE_TORCH_COMPILE: If True, compile the model (ignored when DATAPARALLEL).
        device (torch.device): PyTorch device to place the model on (e.g., 'cuda' or 'cpu')
        verbose: If True, print/log progress information. Defaults to True.

    Returns:
        RGBPOLDecomposer: The initialized model ready for training or inference
        
    Note:
        The model expects input tensors of shape [B×3×H×W] for RGB images and 
        [B×3×H×W] for polarization images, where B is batch size, H and W are \
        height and width respectively.
    """
    # Device by DISTRIBUTE: DDP keeps per-rank device (set by main); DP uses cuda:0 when multi-GPU
    distribute = config.get("DISTRIBUTE", DISTRIBUTE_SINGLEGPU)
    if distribute == DISTRIBUTE_DDP:
        # main.py sets device to cuda:local_rank; do not override
        pass
    elif distribute == DISTRIBUTE_DP and device.type == "cuda" and torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
        if verbose:
            logger.info(
                "DataParallel mode: using primary device cuda:0 and skipping torch.compile",
                context="MODEL",
            )
    # singlegpu: use provided device as-is
    # Access model configuration from the nested structure
    model_config = config.get("MODEL", {})  # .get("value", {})

    # Get the models module name from config (default: "models")
    model_module_name = model_config.get("MODEL_MODULE", "models")
    # Dynamically import the models module
    models_module = importlib.import_module(model_module_name)

    # Get image dimensions from config (check multiple possible locations)
    target_size = None
    for dataset_name in ["SCRREAM", "HOUSECAT6D", "POLARGB"]:
        dataset_config = (
            config.get("DATASETS", {}).get("value", {}).get(dataset_name, {})
        )
        if "TARGET_SIZE" in dataset_config:
            target_size = dataset_config["TARGET_SIZE"]
            break

    if target_size is None:
        target_size = (224, 224)  # Default fallback

    if isinstance(target_size, list):
        target_size = tuple(target_size)

    # # RGB Encoder configuration
    rgb_encoder_config = model_config.get("RGB_ENCODER", {})
    encoder_name = rgb_encoder_config.get(
        "ENCODER", "facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    is_convnext = "convnext" in encoder_name.lower()

    dinov3_cfg = {
        "model_name": encoder_name,
        "image_size": rgb_encoder_config.get("IMAGE_SIZE", min(target_size)),
        "freeze_backbone": rgb_encoder_config.get("FREEZE_BACKBONE", True),
        "return_selected_layers": rgb_encoder_config.get(
            "RETURN_SELECTED_LAYERS", [3, 6, 9, 12]
        ),
        "return_last_hidden_state": rgb_encoder_config.get(
            "RETURN_LAST_HIDDEN_STATE", False
        ),
        "return_as_feature_maps": is_convnext,  # ConvNeXt should return spatial maps
        "return_cls_token": False,
        # Learning rate for RGB encoder - if 0, encoder will be frozen; otherwise sets its learning rate
        "encoder_lr": rgb_encoder_config.get("RGB_ENCODER_LR", None),
    }

    # POL Encoder configuration
    pol_encoder_config = model_config.get("POL_ENCODER", {})
    pol_enc_cfg = {
        "in_ch": 3,
        "embed_dim": pol_encoder_config.get("EMBED_DIM", 384),
        "depth": pol_encoder_config.get("DEPTH", 4),
        "n_heads": pol_encoder_config.get("N_HEADS", 12),
        "patch_size": pol_encoder_config.get("PATCH_SIZE", 16),
    }
    # Cross-attention configuration
    cross_attn_config = model_config.get("CROSS_ATTN", {})
    cross_attn_cfg = {
        "embed_dim": cross_attn_config.get("EMBED_DIM", 384),
        "n_heads": cross_attn_config.get("N_HEADS", 12),
        "dropout": cross_attn_config.get("DROPOUT", 0.1),
        "bi_directional": cross_attn_config.get("BI_DIRECTIONAL", False),
    }

    # Decoder configuration - support both flexible and legacy formats
    decoders_config = model_config.get("DECODERS", None)

    if decoders_config is not None:
        # New flexible decoder format
        decoders = {}
        for decoder_name, decoder_params in decoders_config.items():
            # Build decoder config with defaults
            decoder_cfg = {
                "feature_dim": decoder_params.get("FEATURE_DIM", 384),
                "reassemble_out_channels": decoder_params.get(
                    "REASSEMBLE_OUT_CHANNELS", [12, 24, 48, 92]
                ),
                "reassemble_factors": decoder_params.get(
                    "REASSEMBLE_FACTORS", [4.0, 2.0, 1.0, 0.5]
                ),
                "readout_type": decoder_params.get("READOUT_TYPE", "ignore"),
                "use_bn": decoder_params.get("USE_BN", True),
                "dropout": decoder_params.get("DROPOUT", 0.0),
                "output_image_size": decoder_params.get(
                    "OUTPUT_IMAGE_SIZE", [min(target_size), min(target_size)]
                ),
                "output_channels": decoder_params.get("OUTPUT_CHANNELS", 3),
                # Controls whether a FiLM-conditioned decoder is instantiated
                "use_film": decoder_params.get("USE_FILM", False),
                # Path to pretrained weights - if set and not empty, decoder will be loaded and frozen
                "from_pretrained": decoder_params.get("FROM_PRETRAINED", ""),
                # Learning rate for this decoder - if 0, decoder will be frozen; otherwise sets its learning rate
                "decoder_lr": decoder_params.get("DECODER_LR", None),
            }
            decoders[decoder_name] = decoder_cfg

        decoder_kwargs = {"decoders": decoders}
    else:
        # Legacy decoder format - single DECODER config applied to all three decoders
        decoder_config = model_config.get("DECODER", {})
        decoder_cfg = {
            "feature_dim": decoder_config.get("FEATURE_DIM", 384),
            "reassemble_out_channels": decoder_config.get(
                "REASSEMBLE_OUT_CHANNELS", [12, 24, 48, 92]
            ),
            "reassemble_factors": decoder_config.get(
                "REASSEMBLE_FACTORS", [4.0, 2.0, 1.0, 0.5]
            ),
            "readout_type": decoder_config.get("READOUT_TYPE", "ignore"),
            "use_bn": decoder_config.get("USE_BN", True),
            "output_image_size": decoder_config.get(
                "OUTPUT_IMAGE_SIZE", [min(target_size), min(target_size)]
            ),
            "output_channels": decoder_config.get("OUTPUT_CHANNELS", 3),
        }

        decoder_kwargs = {
            "spec_decoder": decoder_cfg,
            "diffuse_decoder": decoder_cfg,
            "highlight_decoder": decoder_cfg,
        }

    # Use DINOv3_ConvNext if encoder name contains "convnext"
    if is_convnext:
        shared_dinov3 = models_module.DINOv3_ConvNext(dinov3_cfg).to(device)
    else:
        shared_dinov3 = models_module.DINOv3(dinov3_cfg).to(device)
    # Create the main model
    model_class_str = model_config.get("MODEL_CLASS", "RGBPOLDecomposer")
    # Get the model class from the string name
    model_class = getattr(models_module, model_class_str)

    # Build model kwargs based on model type
    model_kwargs = {
        "dinov3": shared_dinov3,
        **decoder_kwargs,
    }

    # Add POL-specific configs only for RGBPOLDecomposer
    if model_class_str == "RGBPOLDecomposer":
        model_kwargs.update(
            {
                "pol_encoder": pol_enc_cfg,
                "pol_cross_attn": cross_attn_cfg,
            }
        )

    # Add TokenInpainter config for UnReflect_Model_TokenInpainter
    if model_class_str == "UnReflect_Model_TokenInpainter":
        token_inpainter_config = model_config.get("TOKEN_INPAINTER", {})
        token_inpainter_cfg = {
            "token_inpainter_class": token_inpainter_config.get(
                "TOKEN_INPAINTER_CLASS", "TokenInpainter"
            ),
            "token_inpainter_module": token_inpainter_config.get(
                "TOKEN_INPAINTER_MODULE", "models"
            ),
            # Path to pretrained weights - if set and not empty, token inpainter will be loaded
            "from_pretrained": token_inpainter_config.get("FROM_PRETRAINED", ""),
            # TokenInpainter parameters (will be passed to the class constructor)
            "depth": token_inpainter_config.get("DEPTH", 4),
            "heads": token_inpainter_config.get("HEADS", 16),
            "drop": token_inpainter_config.get("DROP", 0.1),
            "use_positional_encoding": token_inpainter_config.get(
                "USE_POSITIONAL_ENCODING", True
            ),
            "use_final_norm": token_inpainter_config.get("USE_FINAL_NORM", True),
            "use_local_prior": token_inpainter_config.get("USE_LOCAL_PRIOR", True),
            "local_prior_weight": token_inpainter_config.get("LOCAL_PRIOR_WEIGHT", 0.5),
            "local_prior_kernel": 7,  # token_inpainter_config.get("LOCAL_PRIOR_KERNEL", 5),
            "seed_noise_std": token_inpainter_config.get("SEED_NOISE_STD", 0.01),
        }
        model_kwargs["token_inpainter_cfg"] = token_inpainter_cfg

    model = model_class(**model_kwargs).to(device)
    # torch.compile is incompatible with nn.DataParallel; skip when using DataParallel
    dp_or_ddp = distribute in (DISTRIBUTE_DP, DISTRIBUTE_DDP)
    should_compile = config.get("USE_TORCH_COMPILE", True) and not dp_or_ddp
    if should_compile:
        start_time = time.time()
        model = torch.compile(
            model,
            mode=config.get(
                "COMPILE_MODE", "default"
            ),  # "default", "reduce-overhead", "max-autotune"
            fullgraph=config.get("COMPILE_FULLGRAPH", False),
            dynamic=config.get("COMPILE_DYNAMIC", None),
        )
        end_time = time.time()
        if verbose:
            logger.info(
                f"Torch Compile time: {end_time - start_time:.2f} seconds",
                context="MODEL",
            )
    torch.cuda.empty_cache()
    if verbose:
        logger.info(
            f"Model with class {model.__class__.__name__} created with {sum(p.numel() for p in model.parameters()):,} parameters",
            context="MODEL",
        )

    return model


def wrap_model_for_parallelization(model: torch.nn.Module, config: DotMap):
    """
    When config.DISTRIBUTE == "dp" and CUDA is available: wrap model in DataParallelWrapper
    (so forward receives scatterable args) and nn.DataParallel when multiple GPUs.
    When DISTRIBUTE == "ddp", return model unchanged (main wraps with DDP).
    When DISTRIBUTE == "single", return model unchanged.
    """
    if config.get("DISTRIBUTE") == DISTRIBUTE_DDP:
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.get("LOCAL_RANK")],
            find_unused_parameters=True,
        )
    if config.get("DISTRIBUTE") == DISTRIBUTE_DP:
        model = DataParallelWrapper(model)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return model
    if config.get("DISTRIBUTE") == DISTRIBUTE_SINGLEGPU or not torch.cuda.is_available(): # CPU Only or single gpu
        return model
    


def create_datasets_from_config(
    config: DotMap, dataset_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create training and validation datasets from configuration.

    Reads from YAML config DATASETS section, supports multiple datasets
    (SCRREAM, HOUSECAT6D, POLARGB, etc.), creates dataset-specific classes,
    and returns the Engine-ready dict with ConcatDatasets.

    Logic:
    - VAL_SCENES: defines which scenes to use for validation
    - TRAIN_SCENES: if provided and not None/[], overrides the default training scenes
    - If TRAIN_SCENES is None/[], training uses all scenes except those in VAL_SCENES

    Args:
        config (DotMap): Configuration with DATASETS section and optional WORKERS.
        dataset_names: Optional list of dataset names to load. If None, uses all
            datasets present in config.DATASETS.

    Returns:
        Dict with keys 'Training', 'Validation', 'Test' (ConcatDataset or None),
        and 'workers' (int).
    """

    from dataset.wrappers import (
        SCRREAM_Dataset,
        HOUSECAT6D_Dataset,
        POLARGB_Dataset,
        SCARED_Dataset,
        STEREOMIS_TRACKING_Dataset,
        CHOLEC80_Dataset,
        CROMO_Dataset,
        PSD_Dataset,
        SUNRGBD_Dataset,
    )

    try:
        global_config = config
        if dataset_names is None:
            dataset_names = []
            datasets_config = config.DATASETS
            if isinstance(datasets_config, dict) and datasets_config is not None:
                dataset_names = [
                    name
                    for name in datasets_config.keys()
                    if isinstance(datasets_config[name], dict)
                ]

        if not dataset_names:
            raise ValueError(
                "No datasets found in configuration. Check DATASETS section in config file."
            )

        dataset_classes = {
            "SCRREAM": SCRREAM_Dataset,
            "HOUSECAT6D": HOUSECAT6D_Dataset,
            "POLARGB": POLARGB_Dataset,
            "SCARED": SCARED_Dataset,
            "STEREOMIS_TRACKING": STEREOMIS_TRACKING_Dataset,
            "CHOLEC80": CHOLEC80_Dataset,
            "CROMO": CROMO_Dataset,
            "PSD": PSD_Dataset,
            "SUNRGBD": SUNRGBD_Dataset,
        }

        global_train_scenes = global_config.get("TRAIN_SCENES", {}).get("value")
        global_val_scenes = global_config.get("VAL_SCENES", {}).get("value")

        train_datasets = []
        val_datasets = []

        logger.info(f"Processing {len(dataset_names)} datasets: {dataset_names}")

        for dataset_name in dataset_names:
            if dataset_name not in dataset_classes:
                logger.warning(
                    f"Warning: Dataset class for '{dataset_name}' not found. Skipping."
                )
                logger.info(f"Available classes: {list(dataset_classes.keys())}")
                continue

            datasets_value = global_config.DATASETS
            if datasets_value is None:
                raise ValueError("DATASETS['value'] is None in config")
            dataset_config = datasets_value[dataset_name]

            def get_config_value(param_name, default_value):
                dataset_value = dataset_config.get(param_name)
                if dataset_value is not None:
                    return dataset_value
                global_param = global_config.get(param_name, {})
                if isinstance(global_param, dict) and "value" in global_param:
                    return global_param["value"]
                return default_value

            dataset_params = {
                "rho_s": get_config_value("RHO_S", 0.6),
                "eps": get_config_value("EPS", 1e-8),
                "target_size": tuple(get_config_value("TARGET_SIZE", [224, 224])),
                "resize_mode": get_config_value("RESIZE_MODE", "crop"),
                "use_cache": get_config_value("USE_CACHE", True),
                "simplify_upsampling": get_config_value("SIMPLIFY_UPSAMPLING", True),
                "few_images": get_config_value("FEW_IMAGES", False),
                "sample_every_n": get_config_value("SAMPLE_EVERY_N", 1),
                "load_rgb_only": get_config_value("LOAD_RGB_ONLY", False),
                "highlight_enable": get_config_value("HIGHLIGHT_ENABLE", False),
                "highlight_brightness_threshold": get_config_value(
                    "HIGHLIGHT_BRIGHTNESS_THRESHOLD", 0.93
                ),
                "highlight_return_mask": get_config_value(
                    "HIGHLIGHT_RETURN_MASK", False
                ),
                "highlight_return_rect": get_config_value(
                    "HIGHLIGHT_RETURN_RECT", False
                ),
                "highlight_return_rect_as_rgb": get_config_value(
                    "HIGHLIGHT_RETURN_RECT_AS_RGB", False
                ),
            }

            rect_size_val = get_config_value("HIGHLIGHT_RECT_SIZE", None)
            if rect_size_val is not None:
                try:
                    dataset_params["highlight_rect_size"] = tuple(rect_size_val)
                except Exception:
                    dataset_params["highlight_rect_size"] = None

            dataset_train_scenes = dataset_config.get("TRAIN_SCENES", [])
            dataset_val_scenes = dataset_config.get("VAL_SCENES", [])

            val_scenes = (
                global_val_scenes
                if global_val_scenes is not None
                else dataset_val_scenes
            )

            if global_train_scenes is not None and len(global_train_scenes) > 0:
                train_scenes = global_train_scenes
                logger.info(
                    f"Using global TRAIN_SCENES for {dataset_name}: {train_scenes}"
                )
            elif dataset_train_scenes and len(dataset_train_scenes) > 0:
                train_scenes = dataset_train_scenes
                logger.info(
                    f"Using dataset-specific TRAIN_SCENES for {dataset_name}: {train_scenes}"
                )
            else:
                train_scenes = None
                logger.info(
                    f"Using all scenes except VAL_SCENES for {dataset_name} training"
                )

            dataset_class = dataset_classes[dataset_name]

            if train_scenes is not None and len(train_scenes) > 0:
                dataset_params = {**dataset_params, "highlight_enable": True}
                train_dataset = dataset_class(include=train_scenes, **dataset_params)
                if len(train_dataset) > 0:
                    train_datasets.append(train_dataset)
                    logger.info(
                        f"  ✓ Created training dataset for {dataset_name}: {len(train_dataset)} samples from specific scenes"
                    )
                else:
                    logger.warning(f"  ✗ Training dataset for {dataset_name} is empty")
            else:
                exclude_scenes = (
                    val_scenes if val_scenes and len(val_scenes) > 0 else []
                )
                train_dataset = dataset_class(exclude=exclude_scenes, **dataset_params)
                if len(train_dataset) > 0:
                    train_datasets.append(train_dataset)
                    excluded_text = (
                        f" (excluding {len(exclude_scenes)} val scenes)"
                        if exclude_scenes
                        else ""
                    )
                    logger.info(
                        f"  ✓ Created training dataset for {dataset_name}: {len(train_dataset)} samples{excluded_text}"
                    )
                else:
                    logger.warning(f"  ✗ Training dataset for {dataset_name} is empty")

            if val_scenes and len(val_scenes) > 0:
                dataset_params = {**dataset_params, "highlight_enable": False}
                val_dataset = dataset_class(include=val_scenes, **dataset_params)
                if len(val_dataset) > 0:
                    val_datasets.append(val_dataset)
                    logger.info(
                        f"  ✓ Created validation dataset for {dataset_name}: {len(val_dataset)} samples from {len(val_scenes)} scenes"
                    )
                else:
                    logger.warning(
                        f"  ✗ Validation dataset for {dataset_name} is empty"
                    )
            else:
                logger.warning(f"  ! No validation scenes specified for {dataset_name}")

        training = ConcatDataset(train_datasets) if train_datasets else None
        validation = ConcatDataset(val_datasets) if val_datasets else None

        logger.info("=== Dataset Creation Summary ===")
        logger.info(f"Training:   {len(training) if training else 0} total samples")
        logger.info(f"Validation: {len(validation) if validation else 0} total samples")
        logger.info(f"Test:       {len(validation) if validation else 0} total samples")

        return {
            "Training": training,
            "Validation": validation,
            "Test": validation,
            "workers": config.get("WORKERS", 4),
        }

    except Exception as e:
        logger.error(f"Failed to create datasets using new system: {e}")
        logger.warning("This may be due to:")
        logger.warning("1. Missing or incorrect DATASETS section in config file")
        logger.warning("2. Invalid dataset root directories")
        logger.warning("3. Missing dataset classes")
        logger.warning("Please check your config_train.yaml file structure")
        raise


def load_config_from_checkpoint(checkpoint: dict):
    """Extract and normalize configuration information from a checkpoint."""
    from dotmap import DotMap

    raw_config = checkpoint.get("config")
    if raw_config is None:
        return None
    if isinstance(raw_config, DotMap):
        cfg = raw_config
    elif isinstance(raw_config, dict):
        cfg = DotMap(raw_config)
    else:
        # Some checkpoints may store JSON strings.
        cfg = DotMap(json.loads(raw_config)) if isinstance(raw_config, str) else None
    if cfg is not None:
        cfg.USE_TORCH_COMPILE = False
    return cfg


def load_config_from_path_or_dict(config_path_or_dict):
    """Load DotMap config from a YAML path, a dict, or an existing DotMap."""
    from pathlib import Path
    from dotmap import DotMap
    from utilities.config import load_and_process_config

    if config_path_or_dict is None:
        return None
    if isinstance(config_path_or_dict, DotMap):
        config_path_or_dict.USE_TORCH_COMPILE = False
        return config_path_or_dict
    if isinstance(config_path_or_dict, dict):
        cfg = DotMap(config_path_or_dict)
        cfg.USE_TORCH_COMPILE = False
        return cfg
    path = Path(config_path_or_dict).expanduser().resolve()
    if not path.exists():
        return None
    return load_and_process_config(config_path=str(path))


def load_and_process_config(
    config_path: str,
    config: Optional[Dict[str, Any]] = None,
    unknown_args: Optional[List[str]] = None,
    boot_mode: bool = False,
) -> DotMap:
    """
    Load and process configuration from file or direct input.

    This function handles configuration loading with support for:
    - YAML file parsing with parameter extraction
    - Command-line argument override processing
    - Type-safe parameter conversion
    - Boot mode for quick testing with minimal parameters
    - Automatic dataset configuration updates

    Args:
        config_path (str): Absolute path to the YAML configuration file
        config (Optional[Dict[str, Any]]): Direct configuration dictionary that
                                          overrides file loading if provided
        unknown_args (Optional[List[str]]): List of unknown command-line arguments
                                           to process as parameter overrides
        boot_mode (bool): Whether to enable boot mode with minimal parameters:
                         - BATCH_SIZE=1, EPOCHS=1, NO_WANDB=True
                         - FEW_IMAGES=True for all datasets

    Returns:
        DotMap: Processed configuration object with dot-notation access to parameters

    Raises:
        FileNotFoundError: If config_path does not exist
        yaml.YAMLError: If YAML file parsing fails
        ValueError: If parameter type conversion fails

    Note:
        Command-line arguments should follow the format --PARAMETER=value.
        Boolean parameters accept: true/1/yes for True, false/0/no for False.
        List parameters should be valid Python literals (e.g., "[1,2,3]").
    """
    if config is None:
        # Load the configuration file
        with open(config_path, "r") as f:
            config_yaml = yaml.safe_load(f)
        config_parameters = config_yaml["parameters"]
        config_dict = {
            k: v.get("value") for k, v in config_parameters.items() if v is not None
        }
    else:
        config_dict = config

    # Process the unknown command-line arguments
    if unknown_args:
        additional_args = {}
        for arg in unknown_args:
            if arg.startswith("--"):
                key_value = arg.lstrip("--").split("=", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    additional_args[key.upper()] = value
                else:
                    key = key_value[0]
                    additional_args[key.upper()] = None

        # Override parameters in the configuration with command-line arguments
        for key, value in additional_args.items():
            if key in config_dict:
                orig_value = config_dict[key]
                orig_type = type(orig_value)
                try:
                    if orig_type is bool:
                        if value.lower() in ("true", "1", "yes"):
                            new_value = True
                        elif value.lower() in ("false", "0", "no"):
                            new_value = False
                        else:
                            raise ValueError(f"Cannot parse boolean value: {value}")
                    elif orig_type is list:
                        new_value = ast.literal_eval(value)
                    else:
                        new_value = orig_type(value)
                    config_dict[key] = new_value
                except Exception as e:
                    print(f"Could not convert value for {key}: {value}, error: {e}")
            else:
                print(f"Warning: Unknown parameter {key}")

    # Convert the configuration dictionary to a DotMap for easy access
    config = DotMap(config_dict)

    # Override FEW_IMAGES for all datasets if FEW_IMAGES_OVERRIDE is True
    if getattr(config, "FEW_IMAGES_OVERRIDE", False):
        for dataset_name, dataset_config in config.DATASETS.items():
            if isinstance(dataset_config, dict):
                dataset_config["FEW_IMAGES"] = True
        logger.info("FEW_IMAGES_OVERRIDE enabled - loading few images for all datasets")

    # Override parameters if boot mode is enabled
    if boot_mode:
        config.BATCH_SIZE = 1
        config.EPOCHS = 1
        config.NO_WANDB = True
        config.USE_TORCH_COMPILE = False
        # Set FEW_IMAGES to True for all datasets in boot mode
        for dataset_name, dataset_config in config.DATASETS.items():
            if isinstance(dataset_config, dict):
                dataset_config["FEW_IMAGES"] = True
        logger.info("Boot mode enabled - using minimal parameters for quick testing")

    config = resolve_distribute(config)
    return config
