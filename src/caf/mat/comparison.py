# Built-Ins
import dataclasses
import logging
import pathlib
import re

# Third Party
import caf.toolkit as ctk
import pandas as pd

# Local Imports
from caf.mat import omx_file, ufm_converter

LOG = logging.getLogger(__name__)


class CompareMatrices(ctk.BaseConfig):
    output_name: pathlib.Path
    matrix_a_path: pathlib.Path
    matrix_a_name: str
    matrix_b_path: pathlib.Path
    matrix_b_name: str
    matrix_sector_system_path: ctk.translation.ZoneCorrespondencePath
    cost_matrix_path: pathlib.Path | None = None
    tld_sector_system_path: ctk.translation.ZoneCorrespondencePath | None = None
    bins: list[int] | None = None

    def run(self, saturn_folder: pathlib.Path, output_path: pathlib.Path) -> None:
        LOG.info("Comparing matrices %s and %s", self.matrix_a_path, self.matrix_b_path)

        LOG.info("Reading %s", self.matrix_a_path)
        stacked_matrix_a = read_ufm(self.matrix_a_path, saturn_folder)
        LOG.info("Reading %s", self.matrix_b_path)
        stacked_matrix_b = read_ufm(self.matrix_b_path, saturn_folder)

        if stacked_matrix_a.keys() != stacked_matrix_b.keys():
            raise ValueError("Read in matrices do not contain the same keys")

        matrix_sector_system = self.matrix_sector_system_path.read(
            factors_mandatory=True, generic_column_names=True
        )

        tld_sector_system = None
        cost_matrix = None

        if self.cost_matrix_path is not None:
            LOG.info("Reading %s", self.cost_matrix_path)
            cost_matrix = read_ufm(self.cost_matrix_path, saturn_folder)

            if cost_matrix.keys() != stacked_matrix_a.keys():
                raise ValueError(
                    "Cost matrix does not contain the same keys as the other matrices"
                )

        if self.tld_sector_system_path is not None:
            tld_sector_system = self.tld_sector_system_path.read(
                factors_mandatory=True, generic_column_names=True
            )

        for key in stacked_matrix_a.keys():
            LOG.info("Comparing %s-%s", self.output_name, key)

            with pd.ExcelWriter(
                output_path / f"{self.output_name}_comparison.xlsx",
                if_sheet_exists="replace",
            ) as writer:

                compare_matrix(
                    writer=writer,
                    matrix_a=stacked_matrix_a[key],
                    matrix_a_name=self.matrix_a_name,
                    matrix_b=stacked_matrix_b[key],
                    matrix_b_name=self.matrix_b_name,
                    level=str(key),
                    matrix_sector_system=matrix_sector_system,
                    cost_matrix=cost_matrix[key] if cost_matrix is not None else None,
                    bins=self.bins,
                    tld_sector_system=tld_sector_system,
                )


def compare_matrix(
    writer: pd.ExcelWriter,
    *,
    matrix_a: pd.DataFrame,
    matrix_a_name: str,
    matrix_b: pd.DataFrame,
    matrix_b_name: str,
    level: str,
    matrix_sector_system: pd.DataFrame,
    cost_matrix: pd.DataFrame | None,
    bins: list[int] | None,
    tld_sector_system: pd.DataFrame | None,
) -> None:
    matrix_report_a = ctk.pandas_utils.MatrixReport(
        matrix=matrix_a,
        translation_factors=matrix_sector_system,
        translation_from_col="from",
        translation_to_col="to",
        translation_factors_col="factors",
    )

    matrix_report_b = ctk.pandas_utils.MatrixReport(
        matrix=matrix_b,
        translation_factors=matrix_sector_system,
        translation_from_col="from",
        translation_to_col="to",
        translation_factors_col="factors",
    )

    if cost_matrix is not None and bins is not None:
        matrix_report_a.trip_length_distribution(
            cost_matrix,
            bins,
            sector_zone_lookup=tld_sector_system,
            zone_column="from",
            sector_column="to",
        )
        matrix_report_b.trip_length_distribution(
            cost_matrix,
            bins,
            sector_zone_lookup=tld_sector_system,
            zone_column="from",
            sector_column="to",
        )
    # or would work here because of the above condition, but xor is more explicit
    elif (cost_matrix is not None) ^ (bins is not None):
        raise ValueError("Both cost_matrix and bins must be provided")

    ctk.pandas_utils.compare_matrices_and_output(
        writer,
        matrix_report_a,
        matrix_report_b,
        name_a=matrix_a_name,
        name_b=matrix_b_name,
        label=level,
    )


def read_ufm(
    matrix_path: pathlib.Path, saturn_folder: pathlib.Path
) -> dict[int, pd.DataFrame]:
    converter = ufm_converter.UFMConverter(saturn_folder)
    omx_path = converter.ufm_to_omx(matrix_path)

    omx_reader = omx_file.OMXFile(omx_path)
    output = {}
    for level_name in omx_reader.matrix_levels:
        matched = re.match(r"^l(\d{2})", level_name, flags=re.IGNORECASE)
        if matched is None:
            continue
        level = int(matched.group(1))

        output[level] = omx_reader.get_matrix_level_dataframe(level_name)

    return output


class UFMComparison(ctk.BaseConfig):
    runs: list[CompareMatrices]
    out_path: pathlib.Path
    saturn_folder: pathlib.Path

    def run(self) -> None:
        self.out_path.mkdir(parents=True, exist_ok=True)
        for run in self.runs:
            run.run(self.saturn_folder, self.out_path)

    def log_file_path(self) -> pathlib.Path:
        self.out_path.mkdir(parents=True, exist_ok=True)
        return self.out_path / "comparison.log"
