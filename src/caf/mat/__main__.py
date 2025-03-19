# Built-Ins
import argparse
import logging
import os
import pathlib
import sys

# Third Party
import caf.toolkit as ctk
import pydantic
import tqdm.contrib.logging as tqdm_log

# Local Imports
import caf.mat as mat

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_TRACEBACK = ctk.arguments.getenv_bool("MAT_TRACEBACK", False)


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=mat.__doc__,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="show caf.mat version and exit",
        action="version",
        version=f"{__package__} {ctk.__version__}",
    )

    subparsers = parser.add_subparsers(
        title="caf mat sub-commands",
        description="List of all available sub-commands",
    )

    nrtp_parser = subparsers.add_parser(
        "ufm-comparison",
        help="Compare 2 UFM matrices",
        description="Compare 2 UFM matrices using specified parameters from a config file",
        formatter_class=ctk.arguments.TidyUsageArgumentDefaultsHelpFormatter,
    )
    nrtp_parser.add_argument(
        "config_path",
        type=pathlib.Path,
        help="path to YAML config file containing run parameters",
    )
    nrtp_parser.set_defaults(
        dataclass_parse_func=_config_parse, model=mat.comparison.UFMComparison
    )

    return parser


def _config_parse(
    model: type[mat.comparison.UFMComparison], args: argparse.Namespace
) -> ctk.BaseConfig:
    """Load parameters from config file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with a `config_path` attribute.
    """
    assert issubclass(model, ctk.BaseConfig)
    return model.load_yaml(args.config_path)


def _parse_args() -> mat.comparison.UFMComparison:
    parser = _create_arg_parser()
    args = parser.parse_args(None if len(sys.argv[1:]) > 0 else ["-h"])
    try:
        return args.dataclass_parse_func(args.model, args)
    except (pydantic.ValidationError, FileNotFoundError) as exc:
        if _TRACEBACK:
            raise
        # Switch to raising SystemExit as this doesn't include traceback
        raise SystemExit(str(exc)) from exc


def main():
    """Run the caf ntem module."""
    parameters = _parse_args()

    details = ctk.ToolDetails(
        __package__,
        mat.__version__,  # ntem.__homepage__, ntem.__source_url__
    )
    with ctk.LogHelper(
        __package__, details, console=False, log_file=parameters.log_file_path()
    ) as log:
        tqdm_log.logging_redirect_tqdm([log.logger, log._warning_logger])
        if _LOG_LEVEL.lower() == "debug":
            log.add_console_handler(log_level=logging.DEBUG)
        elif _LOG_LEVEL.lower() == "info":
            log.add_console_handler(log_level=logging.INFO)
        else:
            raise NotImplementedError(
                "The Enviroment constant 'LOG_LEVEL' should"
                " either be set to 'debug' or 'info"
            )

        try:
            parameters.run()

        except (pydantic.ValidationError, FileNotFoundError) as exc:
            if _TRACEBACK:
                raise
            # Switch to raising SystemExit as this doesn't include traceback
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
