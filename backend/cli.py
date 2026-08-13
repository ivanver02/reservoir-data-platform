"""Command line entry points"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.config.settings import configure_data_root
from backend.data.pipeline import run_etl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reservoir-platform")
    parser.add_argument("--data-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("etl", help="run extraction, cleaning and reconciliation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_root is not None:
        configure_data_root(args.data_root)
    if args.command == "etl":
        print(run_etl())


if __name__ == "__main__":
    main()
