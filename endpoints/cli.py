"""Command-line interface for UnReflectAnything.

This module provides the CLI entry points that wrap the Python API.
All CLI commands are thin wrappers that call the corresponding API functions,
ensuring identical behavior between CLI and programmatic usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _config_path() -> Path | None:
    """Path to the single CLI messages config: env UNREFLECTANYTHING_CLI_MESSAGES or cwd/assets/cli_messages.json."""
    import os

    if os.environ.get("UNREFLECTANYTHING_CLI_MESSAGES"):
        p = Path(os.environ["UNREFLECTANYTHING_CLI_MESSAGES"])
        if p.is_file():
            return p
    p = Path.cwd() / "assets" / "cli_messages.json"
    return p if p.is_file() else None


def _print_subcommand_startup_message(subcommand: str | None) -> None:
    """Print optional banner and, when subcommand is set, a short startup line.

    Config: single file from _config_path() (env or cwd/assets/cli_messages.json).
    - "banner": path string (e.g. "project_ascii_banner.txt" next to config) or list of lines.
    - "show_banner_for": subcommand names that get the banner; use "help" to show banner for main --help.
    - Per-subcommand keys (e.g. "train") are the one-line message printed after the banner.
    """
    config_path = _config_path()
    if not config_path:
        return
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    show_banner_for = data.get("show_banner_for") or []
    show_banner = (subcommand is None and "help" in show_banner_for) or (
        subcommand is not None and subcommand in show_banner_for
    )
    if show_banner:
        banner = data.get("banner")
        if banner is not None:
            if isinstance(banner, list):
                for line in banner:
                    print(line, flush=True)
            else:
                config_dir = config_path.parent
                banner_path = (
                    (config_dir / banner)
                    if not Path(banner).is_absolute()
                    else Path(banner)
                )
                if not banner_path.is_file():
                    banner_path = Path.cwd() / banner
                if banner_path.is_file():
                    print(
                        banner_path.read_text(encoding="utf-8"), end="\n\n", flush=True
                    )
    if subcommand is not None:
        msg = data.get(subcommand)
        if isinstance(msg, str) and msg:
            print(msg, flush=True)


# =============================================================================
# CLI Command Handlers - Thin wrappers around API functions
# =============================================================================


def _run_inference(args: argparse.Namespace) -> None:
    """Run inference - calls inference_.inference()."""
    from .inference_ import inference

    # Determine output paths
    output = args.output
    if output is None:
        output = Path("./output")

    inference(
        input=args.input,
        output=output,
        weights_path=args.weights if args.weights else None,
        config=args.config if args.config else None,
        device=args.device,
        batch_size=args.batch_size,
        brightness_threshold=args.brightness_threshold,
        resize_output=not args.no_resize,
        verbose=args.verbose,
    )


def _run_train(args: argparse.Namespace) -> None:
    """Run training - calls train_.train()."""
    from .train_ import train

    # Parse passthrough arguments into overrides dict
    overrides = {}
    passthrough = getattr(args, "passthrough", []) or []

    resume_run = None
    boot = False
    config = args.config

    i = 0
    while i < len(passthrough):
        arg = passthrough[i]
        if arg == "--resume-run" and i + 1 < len(passthrough):
            resume_run = passthrough[i + 1]
            i += 2
            continue
        elif arg == "--boot" or arg == "-b":
            boot = True
            i += 1
            continue
        elif arg.startswith("--config="):
            config = arg.split("=", 1)[1]
            i += 1
            continue
        elif arg == "--config" or arg == "-c":
            if i + 1 < len(passthrough):
                config = passthrough[i + 1]
                i += 2
                continue
        elif arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            overrides[key.upper()] = value
            i += 1
            continue
        i += 1

    train(
        config=config,
        resume_run=resume_run,
        boot=boot,
        **overrides,
    )


def _run_test(args: argparse.Namespace) -> None:
    """Run testing - calls test_.test()."""
    from .test_ import test

    # Parse passthrough arguments into overrides dict
    overrides = {}
    passthrough = getattr(args, "passthrough", []) or []
    config = args.config

    i = 0
    while i < len(passthrough):
        arg = passthrough[i]
        if arg.startswith("--config="):
            config = arg.split("=", 1)[1]
            i += 1
            continue
        elif arg == "--config" or arg == "-c":
            if i + 1 < len(passthrough):
                config = passthrough[i + 1]
                i += 2
                continue
        elif arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            overrides[key.upper()] = value
            i += 1
            continue
        i += 1

    test(config=config, **overrides)


def _run_download(args: argparse.Namespace) -> None:
    """Download assets - calls download_.download()."""
    from .download_ import download

    # Determine what to download
    if args.all:
        what = "all"
    elif args.weights:
        what = "weights"
    elif args.images:
        what = "images"
    elif args.notebooks:
        what = "notebooks"
    elif args.configs:
        what = "configs"
    else:
        # Default to weights if nothing specified
        what = "weights"

    download(
        what=what,
        output_dir=args.output_dir,
        variant=getattr(args, "variant", "default"),
        force=args.force,
    )


def _run_download_weights(args: argparse.Namespace) -> None:
    """Download weights (legacy command) - calls download_.download()."""
    from .download_ import download

    download(
        what="weights",
        output_dir=args.output_dir,
        variant=args.variant,
        force=args.force,
    )


def _run_cache_dir(args: argparse.Namespace) -> None:
    """Print the cache directory (base or a specific asset subdir)."""
    from ._shared import get_cache_dir

    if args.weights:
        path = get_cache_dir("weights")
    elif args.images:
        path = get_cache_dir("images")
    elif args.notebooks:
        path = get_cache_dir("notebooks")
    elif args.configs:
        path = get_cache_dir("configs")
    else:
        path = get_cache_dir()
    print(path.resolve())


def _run_verify(args: argparse.Namespace) -> None:
    """Verify dataset or weights - calls verify_.verify()."""
    from .verify_ import verify

    if args.dataset:
        what = "dataset"
        path = args.path
        if path is None:
            sys.exit("Error: --path is required when using --dataset")
    elif args.weights:
        what = "weights"
        path = None
    else:
        sys.exit("Error: specify either --dataset or --weights")

    is_valid = verify(
        what=what,
        path=path,
        weights_path=args.weights_path,
        dataset_type=args.type,
        config=args.config,
        model_config_path=args.model_config,
    )

    sys.exit(0 if is_valid else 1)


def _run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate model outputs - calls evaluate_.evaluate()."""
    from .evaluate_ import evaluate
    import json as json_module

    # Parse metrics
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]
    elif args.all:
        metrics = None  # None means all metrics

    results = evaluate(
        output=args.output,
        reference=args.reference,
        metrics=metrics,
        mask=args.mask,
    )

    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 40)

    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json_module.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def _run_cite(args: argparse.Namespace) -> None:
    """Print citation - calls cite_.cite()."""
    from .cite_ import cite

    # Determine format
    if args.apa:
        fmt = "apa"
    elif args.mla:
        fmt = "mla"
    elif args.ieee:
        fmt = "ieee"
    elif args.plain:
        fmt = "plain"
    else:
        fmt = "bibtex"

    citation = cite(format=fmt)
    print(citation)


def _run_sweep(args: argparse.Namespace) -> None:
    """Launch a Weights & Biases sweep."""
    import subprocess

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    cmd = ["wandb", "sweep", str(config_path)] + (args.passthrough or [])
    sys.exit(subprocess.run(cmd).returncode)


def _run_agent(args: argparse.Namespace) -> None:
    """Run a W&B sweep agent."""
    import subprocess

    cmd = ["wandb", "agent"] + (args.passthrough or [])
    sys.exit(subprocess.run(cmd).returncode)


def _run_completion(args: argparse.Namespace) -> None:
    """Print shell completion script."""
    try:
        from importlib.resources import files
        from rich import print

        pkg = files("unreflectanything")
    except Exception:
        import importlib.resources

        pkg = importlib.resources.files("unreflectanything")
    shell = (args.shell or "").strip().lower()
    if "zsh" in shell:
        path = pkg / "data" / "unreflect-completion.zsh"
        print(
            """\nRun the following command to load the completion script

[cyan]echo 'source <(unreflectanything completion zsh)' >> ~/.zshrc[/]

[white]It will append the following lines to your ~/.zshrc file:\n
------------------------------------------------------------------------------------------------
""",
            end="",
        )
    else:
        path = pkg / "data" / "unreflect-completion.bash"
        print(
            """\nRun the following command to load the completion script

[cyan]echo 'source <(unreflectanything completion bash)' >> ~/.bashrc[/]

[white]It will append the following lines to your ~/.bashrc file:\n
------------------------------------------------------------------------------------------------
""",
            end="",
        )
    text = path.read_text(encoding="utf-8")
    print(text, end="")
    print(
        "------------------------------------------------------------------------------------------------"
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Entry point for the unreflectanything / unreflect / ura console script."""
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name or "unreflectanything",
        description="UnReflectAnything: remove specular reflections from RGB images.",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", metavar="SUBCOMMAND", required=True
    )

    # -------------------------------------------------------------------------
    # inference
    # -------------------------------------------------------------------------
    p_inf = subparsers.add_parser(
        "inference",
        help="Run inference on image(s) to remove reflections",
        description="Run the UnReflectAnything model to remove specular reflections from images.",
    )
    p_inf.add_argument(
        "input",
        type=str,
        help="Input image file or directory",
    )
    p_inf.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file/directory (default: ./output/)",
    )
    p_inf.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: ~/.cache/unreflectanything/weights/)",
    )
    p_inf.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to inference config YAML file",
    )
    p_inf.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="CUDA device (e.g. cuda, cuda:0, cuda:1) or cpu (default: cuda)",
    )
    p_inf.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    p_inf.add_argument(
        "--brightness-threshold",
        type=float,
        default=0.8,
        help="Brightness threshold for highlight detection (default: 0.8)",
    )
    p_inf.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize output to match original input dimensions",
    )
    p_inf.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    p_inf.set_defaults(func=_run_inference)

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    p_train = subparsers.add_parser(
        "train",
        help="Train the model",
        description="Train the UnReflectAnything model using the specified configuration.",
    )
    p_train.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_train.yaml",
        help="Path to training config YAML (default: config_train.yaml)",
    )
    p_train.add_argument(
        "passthrough",
        nargs="*",
        help="Additional arguments (--resume-run, --boot, --PARAM=value)",
    )
    p_train.set_defaults(func=_run_train)

    # -------------------------------------------------------------------------
    # test
    # -------------------------------------------------------------------------
    p_test = subparsers.add_parser(
        "test",
        help="Test/evaluate a trained model",
        description="Run evaluation on a trained model checkpoint.",
    )
    p_test.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_test.yaml",
        help="Path to test config YAML (default: config_test.yaml)",
    )
    p_test.add_argument(
        "passthrough",
        nargs="*",
        help="Additional arguments (--PARAM=value)",
    )
    p_test.set_defaults(func=_run_test)

    # -------------------------------------------------------------------------
    # download
    # -------------------------------------------------------------------------
    p_dl = subparsers.add_parser(
        "download",
        help="Download pretrained weights, sample images, notebooks, or configs",
        description="Download assets from the HuggingFace repository.",
    )
    p_dl.add_argument(
        "--weights",
        action="store_true",
        help="Download pretrained model weights",
    )
    p_dl.add_argument(
        "--images",
        action="store_true",
        help="Download sample images for testing",
    )
    p_dl.add_argument(
        "--notebooks",
        action="store_true",
        help="Download example Jupyter notebooks",
    )
    p_dl.add_argument(
        "--configs",
        action="store_true",
        help="Download YAML configs (training, inference, etc.)",
    )
    p_dl.add_argument(
        "--all",
        action="store_true",
        help="Download all assets (weights, images, notebooks, configs)",
    )
    p_dl.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ~/.cache/unreflectanything/)",
    )
    p_dl.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Weights variant to download (default: default)",
    )
    p_dl.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    p_dl.set_defaults(func=_run_download)

    # -------------------------------------------------------------------------
    # download-weights (legacy, kept for backwards compatibility)
    # -------------------------------------------------------------------------
    p_dl_weights = subparsers.add_parser(
        "download-weights",
        help="Download pretrained weights (alias for 'download --weights')",
    )
    p_dl_weights.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save weights (default: cache dir)",
    )
    p_dl_weights.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Weights variant to download (default: default)",
    )
    p_dl_weights.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download even if already present",
    )
    p_dl_weights.set_defaults(func=_run_download_weights)

    # -------------------------------------------------------------------------
    # cache-dir
    # -------------------------------------------------------------------------
    p_cache = subparsers.add_parser(
        "cache-dir",
        help="Print the cache directory used for downloaded assets",
        description="Print the base cache directory (~/.cache/unreflectanything or equivalent), or a specific subdir with --weights, --images, --notebooks, or --configs.",
    )
    p_cache.add_argument(
        "--weights",
        action="store_true",
        help="Print weights cache subdir",
    )
    p_cache.add_argument(
        "--images",
        action="store_true",
        help="Print sample images cache subdir",
    )
    p_cache.add_argument(
        "--notebooks",
        action="store_true",
        help="Print notebooks cache subdir",
    )
    p_cache.add_argument(
        "--configs",
        action="store_true",
        help="Print configs cache subdir",
    )
    p_cache.set_defaults(func=_run_cache_dir)

    # -------------------------------------------------------------------------
    # verify (dataset or weights)
    # -------------------------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify dataset structure or weights integrity",
        description="Verify either that a dataset has the correct structure (--dataset) or that weights are downloaded and load into the model with no key alignment errors (--weights).",
    )
    p_verify.add_argument(
        "--dataset",
        action="store_true",
        help="Verify dataset structure; requires --path",
    )
    p_verify.add_argument(
        "--weights",
        action="store_true",
        help="Verify weights file exists and loads into model with no key alignment errors",
    )
    p_verify.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Dataset root directory (required when --dataset)",
    )
    p_verify.add_argument(
        "--weights-path",
        "-w",
        type=str,
        default=None,
        help="Path to weights file (optional when --weights; default: cache)",
    )
    p_verify.add_argument(
        "--type",
        "-t",
        type=str,
        default=None,
        help="Dataset type: SCRREAM, HOUSECAT6D, POLARGB, etc. (auto-detect if not specified; only with --dataset)",
    )
    p_verify.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config file for dataset verification (only with --dataset)",
    )
    p_verify.add_argument(
        "--model-config",
        "-m",
        type=str,
        default=None,
        help="Model config YAML for weights verification if checkpoint has no embedded config (only with --weights)",
    )
    p_verify.set_defaults(func=_run_verify)

    # -------------------------------------------------------------------------
    # evaluate
    # -------------------------------------------------------------------------
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Compute evaluation metrics between output and reference images",
        description="Calculate image quality metrics comparing model outputs to ground truth.",
    )
    p_eval.add_argument(
        "output",
        type=str,
        help="Output image or directory to evaluate",
    )
    p_eval.add_argument(
        "reference",
        type=str,
        help="Reference (ground truth) image or directory",
    )
    p_eval.add_argument(
        "--metrics",
        "-m",
        type=str,
        default=None,
        help="Comma-separated list of metrics: psnr,ssim,mse,deltaE2000,gmsd,dists",
    )
    p_eval.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Compute all available metrics",
    )
    p_eval.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional mask for masked evaluation",
    )
    p_eval.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    p_eval.set_defaults(func=_run_evaluate)

    # -------------------------------------------------------------------------
    # cite
    # -------------------------------------------------------------------------
    p_cite = subparsers.add_parser(
        "cite",
        help="Print citation for UnReflectAnything",
        description="Output the citation for this project in various formats.",
    )
    p_cite.add_argument(
        "--bibtex",
        action="store_true",
        help="Output BibTeX format (default)",
    )
    p_cite.add_argument(
        "--apa",
        action="store_true",
        help="Output APA format",
    )
    p_cite.add_argument(
        "--mla",
        action="store_true",
        help="Output MLA format",
    )
    p_cite.add_argument(
        "--ieee",
        action="store_true",
        help="Output IEEE format",
    )
    p_cite.add_argument(
        "--plain",
        action="store_true",
        help="Output plain text format",
    )
    p_cite.set_defaults(func=_run_cite)

    # -------------------------------------------------------------------------
    # sweep (W&B)
    # -------------------------------------------------------------------------
    p_sweep = subparsers.add_parser(
        "sweep",
        help="Launch a Weights & Biases sweep",
    )
    p_sweep.add_argument(
        "--config",
        type=str,
        default="config_sweep.yaml",
        help="Path to sweep config YAML (default: config_sweep.yaml)",
    )
    p_sweep.add_argument(
        "passthrough",
        nargs="*",
        help="Arguments passed to wandb sweep",
    )
    p_sweep.set_defaults(func=_run_sweep)

    # -------------------------------------------------------------------------
    # agent (W&B)
    # -------------------------------------------------------------------------
    p_agent = subparsers.add_parser(
        "agent",
        help="Run a W&B sweep agent",
    )
    p_agent.add_argument(
        "passthrough",
        nargs="*",
        help="Arguments passed to wandb agent (e.g. sweep ID)",
    )
    p_agent.set_defaults(func=_run_agent)

    # -------------------------------------------------------------------------
    # completion
    # -------------------------------------------------------------------------
    p_comp = subparsers.add_parser(
        "completion",
        help="Print shell completion script",
    )
    p_comp.add_argument(
        "shell",
        nargs="?",
        default="",
        help="Shell: bash or zsh (default: infer from $SHELL)",
    )
    p_comp.set_defaults(func=_run_completion)

    # -------------------------------------------------------------------------
    # Parse and execute
    # -------------------------------------------------------------------------

    # Banner for main help (no subcommand): show before parsing
    if len(sys.argv) <= 2 and (len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help")):
        _print_subcommand_startup_message(None)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    _print_subcommand_startup_message(args.subcommand)
    args.func(args)


if __name__ == "__main__":
    main()
