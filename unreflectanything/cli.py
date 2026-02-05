"""Command-line interface for UnReflectAnything."""

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
    """Print optional banner and, when subcommand is set, a short startup line (stdlib-only, no heavy imports).

    Config: single file from _config_path() (env or cwd/assets/cli_messages.json).
    - "banner": path string (e.g. "project_ascii_banner.txt" next to config) or list of lines.
    - "show_banner_for": subcommand names that get the banner; use "help" to show banner for main --help.
    - Per-subcommand keys (e.g. "train") are the one-line message printed after the banner.
    When subcommand is None (main help), only the banner is considered (if "help" in show_banner_for).
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
                banner_path = (config_dir / banner) if not Path(banner).is_absolute() else Path(banner)
                if not banner_path.is_file():
                    banner_path = Path.cwd() / banner
                if banner_path.is_file():
                    print(banner_path.read_text(encoding="utf-8"), end="\n\n", flush=True)
    if subcommand is not None:
        msg = data.get(subcommand)
        if isinstance(msg, str) and msg:
            print(msg, flush=True)


def _run_train(args: argparse.Namespace) -> None:
    """Dispatch to main.run_pipeline(mode='train') with current argv."""
    # run_pipeline parses sys.argv; replace with [prog, ...train args]
    argv = sys.argv
    sys.argv = [argv[0]] + args.passthrough
    try:
        import main
        main.run_pipeline(mode="train")
    finally:
        sys.argv = argv


def _run_test(args: argparse.Namespace) -> None:
    """Dispatch to main.run_pipeline(mode='test') with current argv."""
    argv = sys.argv
    sys.argv = [argv[0]] + args.passthrough
    try:
        import main
        main.run_pipeline(mode="test")
    finally:
        sys.argv = argv


def _run_inference(args: argparse.Namespace) -> None:
    """Dispatch to inference entry: parse config and run inference."""
    argv = sys.argv
    # inference.parse_cli() only parses --config; do not pass extra args
    sys.argv = [argv[0], "--config", str(Path(args.config).resolve())]
    try:
        import inference
        inference.main()
    finally:
        sys.argv = argv


def _run_sweep(args: argparse.Namespace) -> None:
    """Launch a Weights & Biases sweep."""
    import subprocess
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    cmd = ["wandb", "sweep", str(config_path)] + args.passthrough
    sys.exit(subprocess.run(cmd).returncode)


def _run_agent(args: argparse.Namespace) -> None:
    """Run a W&B sweep agent."""
    import subprocess
    cmd = ["wandb", "agent"] + args.passthrough
    sys.exit(subprocess.run(cmd).returncode)


def _run_completion(args: argparse.Namespace) -> None:
    """Print shell completion script."""
    try:
        from importlib.resources import files
        pkg = files("unreflectanything")
    except Exception:
        # Python < 3.9 fallback
        import importlib.resources
        pkg = importlib.resources.files("unreflectanything")
    shell = (args.shell or "").strip().lower()
    if "zsh" in shell:
        path = pkg / "data" / "unreflect-completion.zsh"
    else:
        path = pkg / "data" / "unreflect-completion.bash"
    text = path.read_text(encoding="utf-8")
    print(text, end="")


def _run_download_weights(args: argparse.Namespace) -> None:
    """Download pretrained weights to cache or specified directory."""
    from unreflectanything.weights import download_weights
    download_weights(
        output_dir=Path(args.output_dir),
        variant=args.variant,
        force=args.force,
    )


def main() -> None:
    """Entry point for the unreflectanything / unreflect / ura console script."""
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name or "unreflectanything",
        description="UnReflectAnything: remove specular reflections from RGB images.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Run training")
    p_train.add_argument("passthrough", nargs="*", help="Arguments passed to train (e.g. --config, --resume-run)")
    p_train.set_defaults(func=_run_train)

    # test
    p_test = subparsers.add_parser("test", help="Run evaluation / testing")
    p_test.add_argument("passthrough", nargs="*", help="Arguments passed to test")
    p_test.set_defaults(func=_run_test)

    # inference
    p_inf = subparsers.add_parser("inference", help="Run inference on an image directory")
    p_inf.add_argument(
        "--config", "-c",
        type=str,
        default="config_inference.yaml",
        help="Path to inference YAML config (default: config_inference.yaml)",
    )
    p_inf.set_defaults(func=_run_inference)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Launch a Weights & Biases sweep")
    p_sweep.add_argument(
        "--config",
        type=str,
        default="config_sweep.yaml",
        help="Path to sweep config YAML (default: config_sweep.yaml)",
    )
    p_sweep.add_argument("passthrough", nargs="*", help="Arguments passed to wandb sweep")
    p_sweep.set_defaults(func=_run_sweep)

    # agent
    p_agent = subparsers.add_parser("agent", help="Run a W&B sweep agent")
    p_agent.add_argument("passthrough", nargs="*", help="Arguments passed to wandb agent (e.g. sweep ID)")
    p_agent.set_defaults(func=_run_agent)

    # completion
    p_comp = subparsers.add_parser("completion", help="Print shell completion script")
    p_comp.add_argument(
        "shell",
        nargs="?",
        default="",
        help="Shell: bash or zsh (default: infer from $SHELL)",
    )
    p_comp.set_defaults(func=_run_completion)

    # download-weights
    p_dl = subparsers.add_parser("download-weights", help="Download pretrained weights")
    p_dl.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save weights (default: cache dir)",
    )
    p_dl.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Weights variant to download (default: default)",
    )
    p_dl.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if already present",
    )
    p_dl.set_defaults(func=_run_download_weights)

    # Banner for main help (no subcommand): show before parsing so it appears before --help output
    if len(sys.argv) <= 2 and (len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help")):
        _print_subcommand_startup_message(None)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)
    _print_subcommand_startup_message(args.subcommand)
    # Resolve output_dir for download-weights
    if args.subcommand == "download-weights" and args.output_dir is None:
        from unreflectanything.weights import get_weights_cache_dir
        args.output_dir = str(get_weights_cache_dir())
    args.func(args)
