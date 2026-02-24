"""
SATURN UFM to CSV
=================

Example code showing how to use caf.mat's :class:`UFMConverter`
to convert SATURN UFM files to / from OMX.
"""

##### IMPORT #####

import argparse
import logging
import pathlib

import caf.toolkit as ctk

from caf.mat import ufm

##### CONSTANTS #####

_NAME = "caf.mat.ufm_to_omx"
LOG = logging.getLogger(_NAME)

##### FUNCTIONS & CLASSES #####


def parse_args() -> tuple[pathlib.Path, pathlib.Path]:
    """Parse command-line arguments and return them

    Returns
    -------
    pathlib.Path
        Path to SATURN executables folder.
    pathlib.Path
        Folder containing UFM files for conversion.
    """

    def directory(value: str) -> pathlib.Path:
        path = pathlib.Path(value)
        if path.is_dir():
            return path.resolve()

        raise NotADirectoryError(path)

    parser = argparse.ArgumentParser(
        description="Converts CUBE .mat files to CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("saturn_path", help="Path to SATURN executable folder", type=directory)
    parser.add_argument(
        "matrix_folder",
        help="Folder containing UFM files for conversion",
        type=directory,
    )
    args = parser.parse_args()

    return args.saturn_path, args.matrix_folder


def main() -> None:
    """Run UFM to OMX for all UFMs in given folder."""
    saturn_path, matrix_folder = parse_args()

    log_file = matrix_folder / f"{_NAME}.log"
    details = ctk.ToolDetails(_NAME, "0.1.0")

    with ctk.LogHelper("caf.mat", details, log_file=log_file):
        # Find all .UFM files inside given folder and convert each to CSVs separately
        matrices = list(matrix_folder.glob("*.ufm"))

        converter = ufm.UFMConverter(saturn_path)

        for i, path in enumerate(matrices, start=1):
            LOG.info("Converting %s to OMX", path.name)
            omx_path = converter.ufm_to_omx(path)
            LOG.info("Written: %s", omx_path)
            LOG.info("Done %s / %s (%s)", i, len(matrices), f"{i / len(matrices):.0%}")


##### MAIN #####
if __name__ == "__main__":
    main()
