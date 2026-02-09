#  MODULES AND DATASET LOADING
from dotenv import load_dotenv

load_dotenv()
import argparse
import os
import time
import torch

import debugpy
from rich.traceback import install

from engine import Engine
from logger import get_logger
from utilities.system_ops import titlescreen
from typing import Dict, Any, Optional

logger = get_logger(__name__).set_context("IMPORT")


def run_pipeline(mode: str = "train", config: Optional[Dict[str, Any]] = None) -> None:
    """
    Common pipeline for train and test modes.

    This is the main entry point that orchestrates the entire machine learning pipeline:
    - Environment setup and device detection
    - Configuration loading and processing
    - Model and dataset instantiation
    - Training or testing execution via Engine
    - Checkpoint management and evaluation

    The pipeline supports both training and testing modes with comprehensive
    logging, debugging capabilities, and optional experiment tracking.

    Args:
        mode (str): Operation mode, either 'train' or 'test'. Defaults to 'train'.
                   - 'train': Runs full training loop with validation and final testing
                   - 'test': Loads best checkpoint and runs evaluation only
        config (Optional[Dict[str, Any]]): Optional configuration dictionary to use
                                          instead of loading from file

    Returns:
        None

    Raises:
        ValueError: If mode is not 'train' or 'test'
        FileNotFoundError: If configuration file is not found
        RuntimeError: If model creation or dataset loading fails

    Note:
        This function handles:
        - CUDA device detection and setup
        - Multi-core CPU detection for data loading
        - Debug server setup if enabled
        - Experiment note management for reproducibility
        - VM auto-shutdown for cloud environments
    """
    # from utilities import *   # <--- this is not allowed within a function. Check if we need something from utilities
    import utilities.engine_initializers as initialize
    from utilities.config import (
        create_model_from_config,
        create_datasets_from_config,
        load_and_process_config,
        maybe_wrap_data_parallel,
    )

    install(show_locals=False)

    # Argparse
    parser = argparse.ArgumentParser(description=f"{mode.capitalize()} the network")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--wait-debugger-attach",
        "-wd",
        action="store_true",
        help="Wait for debugger to attach",
    )
    parser.add_argument(
        "--record",
        "-r",
        action="store_true",
        help="Save the session for comparison",
    )
    parser.add_argument("--stop", "-s", action="store_true", help="Stop VM when done")
    parser.add_argument(
        "--boot",
        "-b",
        action="store_true",
        help="Run in boot mode with minimal parameters",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=f"config_{mode}.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        help="Resume training from an existing run. Provide the run name or run ID.",
    )
    parser.add_argument(
        "--distill-from",
        type=str,
        help="Train using knowledge distillation from a teacher model. Provide the run name or run ID of the teacher model.",
    )

    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()

    if args.debug:
        debug_port = int(os.getenv("DEBUGPY_PORT"))
        # Get machine hostname/IP for remote debugging
        import socket

        hostname = socket.gethostname()
        try:
            # Try to get the actual IP address
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "localhost"

        logger.info(f"Debug mode enabled on port {debug_port}")
        logger.info(
            f"Connect VSCode debugger to: {hostname} ({ip_address}:{debug_port})"
        )
        debugpy.listen(
            ("0.0.0.0", debug_port)
        )  # Listen on all interfaces for remote connections
        if args.wait_debugger_attach:
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()

    # Show title screen if available
    try:
        titlescreen()
    except Exception:
        logger.info("=" * 50, context="INFO")
        logger.info("UnReflectAnything - Reflection Removal Training", context="INFO")
        logger.info("=" * 50, context="INFO")

    logger.info(f"Torch Version: {torch.__version__}")
    logger.info(f"Python Version: {os.sys.version.split()[0]}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")

    # Get CPU info
    try:
        cpu_affinity = os.sched_getaffinity(os.getpid())
        CPU_AFFINITY = len(list(cpu_affinity))
        logger.info(f"Cores available: {CPU_AFFINITY} {sorted(list(cpu_affinity))}")
    except Exception:
        logger.info("Couldn't get CPU affinity", context="INFO")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process configuration
    CONFIG_PATH = args.config
    config = load_and_process_config(
        config_path=CONFIG_PATH,
        config=config,
        unknown_args=unknown,
        boot_mode=args.boot,
    )

    # Run the appropriate function based on mode
    try:
        if mode == "train":
            # Check if we need to resume from an existing run
            if hasattr(args, "resume_run") and args.resume_run:
                logger.info(
                    f"Resuming training from run: {args.resume_run}", context="RESUME"
                )

                # Create model
                model = create_model_from_config(config, DEVICE)
                model = maybe_wrap_data_parallel(model, config)

                # Create datasets for training
                dataset = create_datasets_from_config(config)

                # Initialize engine
                engine = Engine(
                    model=model,  # Pass the created model
                    dataset=dataset,
                    config=config,
                    no_wandb=config.get("NO_WANDB", False),
                    notes=config.get("NOTES", ""),
                )

                # Mark that we will resume to skip directory setup during init
                engine._will_resume = True

                # Resume from the specified run
                if not engine.resume_from_run(args.resume_run):
                    logger.error(
                        "Failed to resume training. Exiting.", context="RESUME"
                    )
                    return

                # Train the model (will start from the correct epoch)
                start_time = time.time()
                engine.trainloop()
                end_time = time.time()
                logger.info(f"Training time: {end_time - start_time:.2f} seconds")

                # Load best model and test
                engine.reinstantiate_model_from_checkpoint()
                engine.test()

            else:
                # Normal training (new run)
                # Create model
                model = create_model_from_config(config, DEVICE)
                model = maybe_wrap_data_parallel(model, config)

                # Create datasets for training
                dataset = create_datasets_from_config(config)

                # Initialize engine
                engine = Engine(
                    model=model,  # Pass the created model
                    dataset=dataset,
                    config=config,
                    no_wandb=config.get("NO_WANDB", False),
                    resume_run_id=args.resume_run,
                    notes=config.get("NOTES", ""),
                )

                # Train the model
                engine.trainloop()

                # Load best model and test
                engine.reinstantiate_model_from_checkpoint()
                engine.test()

        elif mode == "test":
            # Resolve run identifier from config for test resume
            run_identifier = config.get("RUN", None)
            if run_identifier is None or (
                isinstance(run_identifier, str) and len(run_identifier.strip()) == 0
            ):
                logger.error(
                    "RUN name must be provided for test mode (via config or --RUN)"
                )
                return

            # Discover existing run directories without creating anything new
            from utilities.run_resume import get_resume_info

            resume_info = get_resume_info(
                run_identifier, initialize.device_and_directories(config)["runs_dir"]
            )
            if resume_info is None:
                logger.error(f"No existing run matched RUN name: {run_identifier}")
                return

            # Create model and datasets using the current config
            model = create_model_from_config(config, DEVICE)
            model = maybe_wrap_data_parallel(model, config)
            dataset = create_datasets_from_config(config)

            # Initialize engine in resume mode, resuming the existing WandB run
            engine = Engine(
                model=model,
                dataset=dataset,
                config=config,
                notes=config.get("NOTES", ""),
                no_wandb=False,
                resume_run_id=None,  # let initializer resolve RUN by display name
                will_resume=True,
                resume_info=resume_info,
            )

            # Load best model from existing run and test
            engine.reinstantiate_model_from_checkpoint()
            engine.test()

        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        logger.error(f"Error in run_pipeline: {e}")
        raise e
