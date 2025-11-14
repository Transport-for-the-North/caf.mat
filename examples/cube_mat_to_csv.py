# -*- coding: utf-8 -*-
"""
CUBE .mat to CSV
================

Example code showing how to use caf.mat for converting CUBE
.mat files to CSVs.
"""

##### IMPORTS #####

# Built-Ins
import argparse
import logging
import pathlib

# Third Party
import caf.toolkit as ctk

# Local Imports
from caf.mat.cube import CUBEMatConverter
from caf.mat.omx import OMXFile

##### CONSTANTS #####

_NAME = "cube_mat_to_csv"
LOG = logging.getLogger(_NAME)

##### CLASSES & FUNCTIONS #####


def mat_to_csv(
    converter: CUBEMatConverter, mat_path: pathlib.Path, output_folder: pathlib.Path
) -> None:
    """Convert CUBE .mat file to multiple CSVs (one per level).

    Parameters
    ----------
    converter : CUBEMatConverter
        Instance of CUBE .mat converter.
    mat_path : pathlib.Path
        Path to existing CUBE .mat file.
    output_folder : pathlib.Path
        Folder to save outputs to.
    """
    LOG.info("Converting %s to CSVs", mat_path.name)
    omx_path = converter.mat_2_omx(mat_path, output_folder, mat_path.stem)

    omx = OMXFile(omx_path)

    for level, data in omx.get_all():
        out_path = output_folder / f"{mat_path.stem}-{level}.csv"
        data.to_csv(out_path)
        LOG.info("Written: %s", out_path.name)


def parse_args() -> tuple[pathlib.Path, pathlib.Path]:
    """Parse command-line arguments and return them

    Returns
    -------
    pathlib.Path
        Path to CUBE Voyager executable.
    pathlib.Path
        Folder containing CUBE matrices for conversion.
    """
    parser = argparse.ArgumentParser(
        description="Converts CUBE .mat files to CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "voyager_path", help="Path to CUBE Voyager executable", type=pathlib.Path
    )
    parser.add_argument(
        "matrix_folder",
        help="Folder containing CUBE matrices for conversion",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    return args.voyager_path, args.matrix_folder


def main():
    """Main function for running converting CUBE .mat to CSVs."""
    voyager_path, matrix_folder = parse_args()
    converter = CUBEMatConverter(voyager_path)

    if not matrix_folder.is_dir():
        raise NotADirectoryError(matrix_folder)

    output_folder = matrix_folder / "CSVs"
    output_folder.mkdir(exist_ok=True)

    log_file = output_folder / f"{_NAME}.log"
    details = ctk.ToolDetails(_NAME, "0.1.0")

    with ctk.LogHelper(_NAME, details, log_file=log_file):
        # Find all .mat files inside given folder and convert each to CSVs separately
        for path in matrix_folder.glob("*.mat"):
            mat_to_csv(converter, path, output_folder)


if __name__ == "__main__":
    main()
