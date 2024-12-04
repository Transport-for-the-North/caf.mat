# -*- coding: utf-8 -*-
"""
    SATURN .UFM Files to CSVs
    =========================

    Example code showing how to use caf.mat for converting SATURN
    .UFM files to CSVs.
"""

##### IMPORTS #####

# Built-Ins
import argparse
import logging
import pathlib

# Third Party
import caf.toolkit as ctk

# Local Imports
from caf.mat.omx_file import OMXFile
from caf.mat.ufm_converter import UFMConverter

##### CONSTANTS #####

_NAME = "ufm_to_csv"
LOG = logging.getLogger(_NAME)

##### CLASSES & FUNCTIONS #####


def ufm_to_csv(
    converter: UFMConverter, ufm_path: pathlib.Path, output_folder: pathlib.Path
) -> None:
    """Convert SATURN UFM file to multiple CSVs (one per level).

    Parameters
    ----------
    converter
        Instance of SATURN UFM converter.
    ufm_path
        Path to existing UFM file.
    output_folder
        Folder to save outputs to.
    """
    LOG.info("Converting %s to CSVs", ufm_path.name)
    omx_path = converter.ufm_to_omx(ufm_path, output_folder, ufm_path.stem)

    omx = OMXFile(omx_path)

    for level in omx.matrix_levels:
        data = omx.get_matrix_level_dataframe(level)
        out_path = output_folder / f"{ufm_path.stem}-{level}.csv"
        data.to_csv(out_path)
        LOG.info("Written: %s", out_path.name)


def parse_args() -> tuple[pathlib.Path, pathlib.Path]:
    """Parse command-line arguments and return them

    Returns
    -------
    pathlib.Path
        Path to SATURN folder containing batch files.
    pathlib.Path
        Folder containing UFM matrices for conversion.
    """
    parser = argparse.ArgumentParser(
        description="Converts SATURN UFM files to CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("saturn_path", help="Path to SATURN folder", type=pathlib.Path)
    parser.add_argument(
        "matrix_folder",
        help="Folder containing CUBE matrices for conversion",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    return args.saturn_path, args.matrix_folder


def main():
    """Main function for running converting CUBE .mat to CSVs."""
    saturn_path, matrix_folder = parse_args()
    converter = UFMConverter(saturn_path)

    if not matrix_folder.is_dir():
        raise NotADirectoryError(matrix_folder)

    output_folder = matrix_folder / "CSVs"
    output_folder.mkdir(exist_ok=True)

    log_file = output_folder / f"{_NAME}.log"
    details = ctk.ToolDetails(_NAME, "0.1.0")

    with ctk.LogHelper(_NAME, details, log_file=log_file):
        # Find all .mat files inside given folder and convert each to CSVs separately
        for path in matrix_folder.glob("*.ufm"):
            ufm_to_csv(converter, path, output_folder)


if __name__ == "__main__":
    main()
