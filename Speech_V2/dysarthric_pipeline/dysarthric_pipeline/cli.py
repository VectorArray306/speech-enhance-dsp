"""Command line entry point for the dysarthric speech pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from dysarthric_pipeline import __version__
from dysarthric_pipeline.config import load_config
from dysarthric_pipeline.environment import lexicon_status, summarize_data_root, torch_status


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dysarthric-pipeline",
        description="Dysarthric speech to text and voice.",
    )
    parser.add_argument("--config", default=None, help="YAML config. Defaults to configs/default.yaml.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Verify data root, word codes, and CUDA.")
    check.add_argument("--data-root", default=None, help="Override paths.data_root for this run.")

    commands.add_parser("prepare", help="Build manifests (phase 2).")
    train = commands.add_parser("train", help="Fine-tune the general ASR adapter.")
    train.add_argument("--baseline-only", action="store_true", help="Score the pretrained model and stop.")
    train.add_argument("--skip-baseline", action="store_true", help="Skip the zero-shot score.")
    train.add_argument("--epochs", type=int, default=None)
    adapt = commands.add_parser("adapt", help="Fit a speaker adapter from the adapt split or a wav/txt folder.")
    adapt.add_argument("--speaker", default=None)
    adapt.add_argument("--calibration-dir", default=None)
    run = commands.add_parser("run", help="Listen to audio and print text. Optionally speak the text.")
    run.add_argument("--source", choices=("file", "mic", "serial"), default="file")
    run.add_argument("--input", default=None, help="WAV file. Required when --source file.")
    run.add_argument(
        "--device",
        default=None,
        help="Microphone name when --source mic. Example: 'USB AUDIO DEMO'.",
    )
    run.add_argument("--speaker", default=None, help="Use a saved speaker adapter, for example uaspeech:M08.")
    run.add_argument("--output", choices=("text", "voice", "both"), default=None)
    run.add_argument("--play", action="store_true", help="Play the preset voice on the speakers.")
    run.add_argument("--save-dir", default=None, help="Folder for the spoken WAV files.")
    run.add_argument(
        "--no-vad",
        action="store_true",
        help="With --source file, transcribe the whole clip as one utterance.",
    )
    return parser


def _cmd_check(config: dict, args: argparse.Namespace) -> int:
    data_root = Path(args.data_root or config["paths"]["data_root"])
    summary = summarize_data_root(data_root)
    lexicon = lexicon_status()
    gpu = torch_status()
    print(f"config: {config['_config_path']}")
    print(f"data root: {summary['root']}")
    for group, speakers in summary["groups"].items():
        print(f"  TORGO {group}: {', '.join(speakers) if speakers else '(missing)'}")
    raw = summary["rawdyspeech"]
    print(
        "  RAWDysPeech: "
        + ", ".join(f"{label}={count}" for label, count in raw.items())
        if raw
        else "  RAWDysPeech: (missing)"
    )
    print(
        f"lexicon: {lexicon['closed_set']} closed-set words; "
        f"open-class probes missing: {lexicon['pending_open_classes']}"
    )
    if gpu["installed"] and gpu["cuda"]:
        print(f"torch: {gpu['version']} on {gpu['device']}")
    elif gpu["installed"]:
        print(f"torch: {gpu['version']}, CUDA not visible")
    else:
        print("torch: not installed in this interpreter")
    print("phase 1 check passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "check":
        return _cmd_check(config, args)
    if args.command == "prepare":
        from dysarthric_pipeline.data.prepare_dataset import build_manifests

        build_manifests(config)
        return 0
    if args.command == "train":
        from dysarthric_pipeline.models.train_asr import train

        if args.epochs is not None:
            config["train"]["epochs"] = args.epochs
        train(config, baseline_only=args.baseline_only, skip_baseline=args.skip_baseline)
        return 0
    if args.command == "adapt":
        from dysarthric_pipeline.models.adapt_speaker import adapt

        adapt(config, args.speaker, args.calibration_dir)
        return 0
    if args.command == "run":
        from dysarthric_pipeline.io_streams.runtime import open_source, run_session

        if args.no_vad and args.source != "file":
            parser.error("--no-vad is only valid with --source file")
        source = open_source(config, args.source, args.input, device=getattr(args, "device", None))
        output_mode = args.output or config.get("output", {}).get("mode", "text")
        run_session(
            config,
            source,
            speaker=args.speaker,
            output_mode=output_mode,
            play=args.play,
            save_dir=args.save_dir,
            segment=not args.no_vad,
        )
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
