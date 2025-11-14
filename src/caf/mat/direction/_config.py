# -*- coding: utf-8 -*-
"""Configuration files and parameters for running caf.mat direction functionality."""

##### IMPORTS #####

# Built-Ins
import argparse
import datetime
import logging
import pathlib

# Third Party
import caf.toolkit as ctk
import pydantic

# Local Imports
from caf.mat import _mat
from caf.mat.direction import od_to_pa

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class _OD2PAArguments(_mat.ArgumentHandler):
    """Define arguments for OD to PA sub-command."""

    config: pydantic.FilePath = pydantic.Field(description="path to config file for od2pa")

    def run(self) -> None:
        """Run OD to PA conversion."""
        config = od_to_pa.OD2PAParameters.load_yaml(self.config)
        return od_to_pa.main(config)

    @property
    def log_path(self) -> pathlib.Path:
        """Define log file path for OD to PA conversion."""
        # False positive caused by pydantic.Field
        # pylint: disable=no-member
        return self.config.parent / f"od2pa-{datetime.date.today():%Y%m%d}.log"


def add_direction_commands(parser: argparse.ArgumentParser) -> None:
    """Add sub-commandas to the parser for direction functionality."""
    subparsers = parser.add_subparsers(
        title="Direction Sub-Commands",
        description="List of available direction conversion commands",
    )

    od2pa_args = ctk.arguments.ModelArguments(_OD2PAArguments)
    od2pa_args.add_subcommands(
        subparsers,
        "od2pa",
        add_config=False,
        help="convert an OD matrix to PA",
        description="Convert a transport demand matrix from OD to PA format.",
        formatter_class=ctk.arguments.TidyUsageArgumentDefaultsHelpFormatter,
    )
