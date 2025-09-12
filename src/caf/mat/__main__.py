# -*- coding: utf-8 -*-
"""Front-end module for running caf.mat functionality from command-line."""

##### IMPORTS #####


# Built-Ins
import argparse
import logging
import sys

# Third Party
import caf.toolkit as ctk
import pydantic

# Local Imports
from caf import mat
from caf.mat import _mat
from caf.mat.direction import _config

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_TRACEBACK = ctk.arguments.getenv_bool("CAF_MAT_TRACEBACK", True)

##### CLASSES & FUNCTIONS #####


def _create_arg_parser() -> argparse.ArgumentParser:
    """Create ArgumentParser with all caf.mat sub-commands."""
    parser = argparse.ArgumentParser(
        __package__,
        description=mat.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="show caf.mat version and exit",
        action="version",
        version=f"{__package__} {mat.__version__}",
    )

    subparsers = parser.add_subparsers(
        title="CAF.mat sub-commands",
        description="List of all available sub-commands",
    )

    direction_parser = subparsers.add_parser(
        "direction",
        help="perform matrix direction conversions",
        description="Convert between PA and OD matrix formats.",
    )
    _config.add_direction_commands(direction_parser)

    return parser


def parse_args() -> _mat.ArgumentHandler:
    """Parse and validate command-line arguments."""
    parser = _create_arg_parser()

    # Print help if no arguments are given
    args = parser.parse_args(None if len(sys.argv[1:]) > 0 else ["-h"])

    try:
        params = args.dataclass_parse_func(args)
    except (pydantic.ValidationError, FileNotFoundError) as exc:
        if _TRACEBACK:
            raise
        # Switch to raising SystemExit as this doesn't include traceback
        raise SystemExit(str(exc)) from exc

    return params


def main():
    """Parser command-line arguments and run CAF.mat functionality."""
    args = parse_args()
    details = ctk.ToolDetails(__package__, mat.__version__)

    with ctk.LogHelper(__package__, details, log_file=args.log_path):
        try:
            args.run()
        except Exception as exc:
            if _TRACEBACK:
                raise
            # Switch to raising SystemExit as this doesn't include traceback
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
