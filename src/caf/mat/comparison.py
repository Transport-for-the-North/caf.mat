# Built-Ins
import abc
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


@dataclasses.dataclass
class UFMInput:
    matrix_path: pathlib.Path
    tp_factor: float


@dataclasses.dataclass
class Comparison(abc.ABC):
    output_name: pathlib.Path
    matrix_a_name: str
    matrix_b_name: str
    matrix_sector_system_path: ctk.translation.ZoneCorrespondencePath
    cost_matrix_path: pathlib.Path | None = None
    tld_sector_system_path: ctk.translation.ZoneCorrespondencePath | None = None
    bins: list[int] | None = None

    @abc.abstractmethod
    def extract_matrix(
        self, saturn_folder: pathlib.Path, matrix_path: pathlib.Path | list[UFMInput]
    ) -> dict[int, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def process_demand_matrices(
        self, saturn_folder: pathlib.Path
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        pass

    def run(self, saturn_folder: pathlib.Path, output_path: pathlib.Path) -> None:

        stacked_matrix_a, stacked_matrix_b = self.process_demand_matrices(saturn_folder)

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

        with pd.ExcelWriter(output_path / f"{self.output_name}_comparison.xlsx") as writer:

            for key in stacked_matrix_a.keys():

                LOG.info("Comparing %s-%s", self.output_name, key)
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


@dataclasses.dataclass(kw_only=True)
class CompareMatrices(Comparison):
    matrix_a_path: pathlib.Path
    matrix_b_path: pathlib.Path

    def process_demand_matrices(
        self, saturn_folder: pathlib.Path
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        LOG.info("Comparing matrices %s and %s", self.matrix_a_path, self.matrix_b_path)
        matrix_a = self.extract_matrix(saturn_folder, self.matrix_a_path)
        matrix_b = self.extract_matrix(saturn_folder, self.matrix_b_path)

        if matrix_a.keys() != matrix_b.keys():
            raise ValueError("Read in matrices do not contain the same keys")

        return matrix_a, matrix_b

    def extract_matrix(
        self, saturn_path: pathlib.Path, matrix_path: pathlib.Path
    ) -> dict[int, pd.DataFrame]:
        LOG.info("Reading %s", matrix_path)
        return read_ufm(matrix_path, saturn_path)


@dataclasses.dataclass(kw_only=True)
class CompareDays(Comparison):
    matrix_a_paths: list[UFMInput]
    matrix_b_paths: list[UFMInput]

    def extract_matrix(
        self, saturn_folder: pathlib.Path, matrix_paths: list[UFMInput]
    ) -> dict[int, pd.DataFrame]:
        matrices: dict[int, pd.DataFrame] = {}
        for ufm in matrix_paths:
            LOG.info("Reading %s", ufm.matrix_path)
            for uc, matrix in read_ufm(ufm.matrix_path, saturn_folder).items():
        
                matrices[uc] = matrix * ufm.tp_factor + matrices.get(uc, 0)
                LOG.debug(
                    "adding to %s with tp factor %s: new total trips %s",
                    uc,
                    ufm.tp_factor,
                    matrices[uc].sum().sum(),
                )

        return matrices

    def process_demand_matrices(
        self, saturn_folder: pathlib.Path
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        matrix_a = self.extract_matrix(saturn_folder, self.matrix_a_paths)
        matrix_b = self.extract_matrix(saturn_folder, self.matrix_b_paths)
        if matrix_a.keys() != matrix_b.keys():
            raise ValueError("Read in matrices do not contain the same keys")

        return matrix_a, matrix_b


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
    out_path: pathlib.Path
    saturn_folder: pathlib.Path
    ufm_comparisons: list[CompareMatrices] | None = None
    day_comparisons: list[CompareDays] | None = None

    def run(self) -> None:
        self.out_path.mkdir(parents=True, exist_ok=True)
        runs: list[Comparison] = []
        if self.ufm_comparisons is not None:
            runs.extend(self.ufm_comparisons)
        if self.day_comparisons is not None:
            runs.extend(self.day_comparisons)
        for r in runs:
            r.run(self.saturn_folder, self.out_path)

    def log_file_path(self) -> pathlib.Path:
        self.out_path.mkdir(parents=True, exist_ok=True)
        return self.out_path / "comparison.log"
