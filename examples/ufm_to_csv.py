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
from caf.mat.ufm import UFMConverter

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
        Folder containing UFM matrices for conversion.
    """
    parser = argparse.ArgumentParser(
        description="Converts SATURN UFM files to CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("saturn_path", help="Path to SATURN folder", type=pathlib.Path)
    parser.add_argument(
        "matrix_folder",
        help="Folder containing UFMs for conversion",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    return args.saturn_path, args.matrix_folder


def main():
    """Main function for running converting SATURN UFMs to CSVs."""
    saturn_path, matrix_folder = parse_args()
    converter = UFMConverter(saturn_path)

    if not matrix_folder.is_dir():
        raise NotADirectoryError(matrix_folder)

    output_folder = matrix_folder / "CSVs"
    output_folder.mkdir(exist_ok=True)

    log_file = output_folder / f"{_NAME}.log"
    details = ctk.ToolDetails(_NAME, "0.1.0")

    with ctk.LogHelper("caf.mat", details, log_file=log_file):
        # Find all .UFM files inside given folder and convert each to CSVs separately
        matrices = list(matrix_folder.glob("*.ufm"))

        for i, path in enumerate(matrices, start=1):
            stacked, unstacked = converter.ufm_to_square_csvs(
                path, path.stem, decimal_places=8
            )

            stacked.unlink()
            LOG.info("Deleted stacked matrix: %s", stacked.name)

            LOG.info(
                "Moving unstacked (%s) files to %s",
                len(unstacked),
                output_folder,
            )
            for mat_path in unstacked:
                mat_path.rename(output_folder / mat_path.name)
                LOG.debug("Moved %s to %s", mat_path.name, output_folder)

            LOG.info("Done %s / %s (%s)", i, len(matrices), f"{i / len(matrices):.0%}")


if __name__ == "__main__":
    main()
