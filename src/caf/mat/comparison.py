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
class BaseComparison(abc.ABC):
    """base class for comparing UFM matrices"""

    output_name: str
    """Name to label the output file."""
    matrix_a_name: str
    """Name to label the first matrix in the comparison."""
    matrix_b_name: str
    """Name to label the second matrix in the comparison."""
    matrix_sector_system_path: ctk.translation.ZoneCorrespondencePath
    """Path to the matrix sector system."""
    cost_matrix_path: pathlib.Path | None = None
    """Path to the cost matrix."""
    tld_sector_system_path: ctk.translation.ZoneCorrespondencePath | None = None
    """Path to the TLD sector system."""
    bins: list[int] | None = None
    """Bins for the trip length distribution."""

    @abc.abstractmethod
    def _extract_matrix(
        self, saturn_folder: pathlib.Path, matrix_path: pathlib.Path | list[UFMInput]
    ) -> dict[int, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def _process_demand_matrices(
        self, saturn_folder: pathlib.Path
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        pass

    def run(self, saturn_folder: pathlib.Path, output_path: pathlib.Path) -> None:
        """Run the comparison between matrices a and b.

        Parameters
        ----------
        saturn_folder : pathlib.Path
            Path to the saturn folder that will be used to convert the UFMs to OMX before reading in.
        output_path : pathlib.Path
            Path to the output directory.
        """

        stacked_matrix_a, stacked_matrix_b = self._process_demand_matrices(saturn_folder)

        matrix_sector_system = self.matrix_sector_system_path.read(
            factors_mandatory=True, generic_column_names=True
        )

        tld_sector_system = None
        cost_matrix = None

        if self.cost_matrix_path is not None:
            LOG.info("Reading %s", self.cost_matrix_path)
            cost_matrix = pd.read_csv(self.cost_matrix_path, index_col=0)
            # TODO(kf): change this to handle UFM too.
            # if cost_matrix.keys() != stacked_matrix_a.keys():
            #    raise ValueError(
            #        "Cost matrix does not contain the same keys as the other matrices"
            #    )
            try:
                cost_matrix.columns = [int(col) for col in cost_matrix.columns]
            except ValueError:
                pass

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
                    cost_matrix=cost_matrix if cost_matrix is not None else None,
                    bins=self.bins,
                    tld_sector_system=tld_sector_system,
                )


@dataclasses.dataclass(kw_only=True)
class CompareMatrices(BaseComparison):
    matrix_a_path: pathlib.Path | dict[int, pathlib.Path]
    """Path to the first matrix in the comparison. 
    If a Path is provided, it should point to a UFM.
    If a dict is provided, the keys are used as levels 
    and paths should point to csvs containing square matrices."""
    matrix_b_path: pathlib.Path | dict[int, pathlib.Path]
    """Path to the second matrix in the comparison.
    If a Path is provided, it should point to a UFM.
    If a dict is provided, the keys are used as levels 
    and paths should point to csvs containing square matrices."""
    levels: list[int] | None = None
    """"Levels to compare.
    If None, all levels in the matrix are used."""

    def _process_demand_matrices(
        self, saturn_folder: pathlib.Path
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        LOG.info("Comparing matrices %s and %s", self.matrix_a_path, self.matrix_b_path)

        matrix_a = self._extract_matrix(saturn_folder, self.matrix_a_path, self.levels)

        matrix_b = self._extract_matrix(saturn_folder, self.matrix_b_path, self.levels)

        if matrix_a.keys() != matrix_b.keys():
            raise ValueError("Read in matrices do not contain the same keys")

        return matrix_a, matrix_b

    def _extract_matrix(
        self,
        saturn_path: pathlib.Path,
        matrix_path: pathlib.Path | dict[int, pathlib.Path],
        levels: list[int] | None = None,
    ) -> dict[int, pd.DataFrame]:
        LOG.info("Reading %s", matrix_path)

        if isinstance(matrix_path, pathlib.Path):
            return ufm_converter.read_ufm(matrix_path, saturn_path, levels)

        elif isinstance(matrix_path, dict):
            if levels is not None:
                if set(levels).issubset(matrix_path.keys()):
                    keys = levels
                else:
                    raise ValueError(
                        f"Levels {levels} are not a subset of the keys {matrix_path.keys()}"
                    )
            else:
                keys = matrix_path.keys()
            matrices: dict[int, pd.DataFrame] = {}
            for k in keys:
                matrices[k] = pd.read_csv(matrix_path[k], index_col=0)
                try:
                    matrices[k].columns = [int(col) for col in matrices[k].columns]
                except ValueError:
                    pass
            return matrices
        else:
            raise ValueError("matrix_path must be a Path or a dict[int, Path]")


@dataclasses.dataclass(kw_only=True)
class CompareDays(BaseComparison):
    matrix_a_paths: list[UFMInput]
    """Matrices and factors to build the first day matrix in the comparison."""
    matrix_b_paths: list[UFMInput]
    """Matrices and factors to build the second day matrix in the comparison."""

    def _extract_matrix(
        self, saturn_folder: pathlib.Path, matrix_paths: list[UFMInput]
    ) -> dict[int, pd.DataFrame]:
        matrices: dict[int, pd.DataFrame] = {}
        for ufm in matrix_paths:
            LOG.info("Reading %s", ufm.matrix_path)
            for uc, matrix in ufm_converter.read_ufm(ufm.matrix_path, saturn_folder).items():

                matrices[uc] = matrix * ufm.tp_factor + matrices.get(uc, 0)
                LOG.debug(
                    "adding to %s with tp factor %s: new total trips %s",
                    uc,
                    ufm.tp_factor,
                    matrices[uc].sum().sum(),
                )

        return matrices

    def _process_demand_matrices(
        self, saturn_folder: pathlib.Path
    ) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
        matrix_a = self._extract_matrix(saturn_folder, self.matrix_a_paths)
        matrix_b = self._extract_matrix(saturn_folder, self.matrix_b_paths)
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
    """Compare two square matrices and output the results to an Excel file.

    Parameters
    ----------
    writer : pd.ExcelWriter
        Excel writer object to write the output to.
    matrix_a : pd.DataFrame
        first matrix to compare.
    matrix_a_name : str
        Name to label the first matrix in the comparison.
    matrix_b : pd.DataFrame
        second matrix to compare.
    matrix_b_name : str
        Name to label the second matrix in the comparison.
    level : str
        Level (usually user-class) of the matrices being compared.
    matrix_sector_system : pd.DataFrame
        Matrix sector system to translate the matrices.
    cost_matrix : pd.DataFrame | None
        Cost matrix to calculate trip length distribution.
    bins : list[int] | None
        Bins for the trip length distribution.
    tld_sector_system : pd.DataFrame | None
        TLD sector system to translate the matrices.
    """
    matrix_report_a = ctk.pandas_utils.MatrixReport(
        matrix=matrix_a.sort_index(axis=1).sort_index(axis=0),
        translation_factors=matrix_sector_system,
        translation_from_col="from",
        translation_to_col="to",
        translation_factors_col="factors",
    )

    matrix_report_b = ctk.pandas_utils.MatrixReport(
        matrix=matrix_b.sort_index(axis=1).sort_index(axis=0),
        translation_factors=matrix_sector_system,
        translation_from_col="from",
        translation_to_col="to",
        translation_factors_col="factors",
    )

    if cost_matrix is not None:
        matrix_report_a.calc_vehicle_kms(cost_matrix=cost_matrix.sort_index(axis=1).sort_index(axis=0),sector_zone_lookup=tld_sector_system,
                zone_column="from",
                sector_column="to",
            )
        matrix_report_b.calc_vehicle_kms(cost_matrix.sort_index(axis=1).sort_index(axis=0),sector_zone_lookup=tld_sector_system,
                zone_column="from",
                sector_column="to",
            )
        if bins is not None:
            matrix_report_a.trip_length_distribution(
                cost_matrix.sort_index(axis=1).sort_index(axis=0),
                bins,
                sector_zone_lookup=tld_sector_system,
                zone_column="from",
                sector_column="to",
            )
            matrix_report_b.trip_length_distribution(
                cost_matrix.sort_index(axis=1).sort_index(axis=0),
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


class UFMComparison(ctk.BaseConfig):
    """Configuration for comparing UFM matrices."""

    out_path: pathlib.Path
    """Path to the output directory."""
    saturn_folder: pathlib.Path
    """Path to the saturn folder. Be mindful of the version of SATURN being used."""
    ufm_comparisons: list[CompareMatrices] | None = None
    """Comparisons between two UFM matrices."""
    day_comparisons: list[CompareDays] | None = None
    """Comparison between two 24hr matrices, built from time period split UFMs."""

    def run(self) -> None:
        """Run the UFM comparison functionaliy."""
        self.out_path.mkdir(parents=True, exist_ok=True)
        runs: list[BaseComparison] = []
        if self.ufm_comparisons is not None:
            runs.extend(self.ufm_comparisons)
        if self.day_comparisons is not None:
            runs.extend(self.day_comparisons)
        for r in runs:
            r.run(self.saturn_folder, self.out_path)

    def log_file_path(self) -> pathlib.Path:
        """Path to the log file for the comparison."""
        self.out_path.mkdir(parents=True, exist_ok=True)
        return self.out_path / "comparison.log"
