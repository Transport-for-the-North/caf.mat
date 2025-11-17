# -*- coding: utf-8 -*-
"""
CSVs to SATURN .UFM Files
=========================

Example code showing how to use caf.mat for converting CSVs
to SATURN .UFM files.
"""

##### IMPORTS #####

# Built-Ins
import argparse
import logging
import pathlib

# Third Party
import caf.toolkit as ctk

# Local Imports
from caf.mat.ufm import CSVFormat, UFMConverter

##### CONSTANTS #####

_NAME = "caf.mat.ufm_to_csv"
LOG = logging.getLogger(_NAME)

##### CLASSES & FUNCTIONS #####


def parse_args() -> tuple[pathlib.Path, pathlib.Path]:
    """Parse command-line arguments and return them

    Returns
    -------
    pathlib.Path
        Path to SATURN folder containing batch files.
    pathlib.Path
        Folder containing CSV matrices for conversion.
    """
    parser = argparse.ArgumentParser(
        description="Converts CSVs to SATURN UFM files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("saturn_path", help="Path to SATURN folder", type=pathlib.Path)
    parser.add_argument(
        "matrix_folder",
        help="Folder containing CSVs for conversion",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    return args.saturn_path, args.matrix_folder


def main():
    """Main function for running converting CSVs to SATURN UFMs."""
    saturn_path, matrix_folder = parse_args()

    if not matrix_folder.is_dir():
        raise NotADirectoryError(matrix_folder)

    output_folder = matrix_folder / "UFMs"
    output_folder.mkdir(exist_ok=True)

    log_file = output_folder / f"{_NAME}.log"
    details = ctk.ToolDetails(_NAME, "0.1.0")

    with ctk.LogHelper("caf.mat", details, log_file=log_file):
        converter = UFMConverter(saturn_path)

        # Find all CSV files inside given folder and convert each to UFMs separately
        matrices = list(matrix_folder.glob("*.csv"))

        for i, path in enumerate(matrices, start=1):
            LOG.info("Converting CSV %s: %s", i, path)
            matrix = ctk.io.read_csv_matrix(path, format_="square")

            # Write CSV in format expected by converter
            csv_path = output_folder / f"{path.stem}-square.csv"
            matrix.to_csv(csv_path, index=True, header=False, float_format="%.10f")
            LOG.debug("Written CSV in SATURNs square format to: %s", csv_path)

            ufm_path = converter.csv_to_ufm(csv_path, CSVFormat.SQUARE)
            LOG.info("Written UFM: %s", ufm_path)

            LOG.info("Done %s / %s (%s)", i, len(matrices), f"{i / len(matrices):.0%}")


if __name__ == "__main__":
    main()
