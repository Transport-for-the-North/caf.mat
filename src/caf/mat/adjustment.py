# Built-Ins
import dataclasses
import logging
import pathlib

# Third Party
import caf.toolkit as ctk
import pandas as pd

# Local Imports
from caf.mat import ufm_converter

LOG = logging.getLogger(__name__)

CHECK_DP_RESOLUTION = 6


@dataclasses.dataclass
class MatrixSectorScaling:
    name: str
    adjustment_ufm: pathlib.Path
    control_ufm: pathlib.Path
    sector_system_path: ctk.translation.ZoneCorrespondencePath
    levels: list[int] | None = None
    exclude_intras: bool = False

    def run(
        self, saturn_path: pathlib.Path, out_path: pathlib.Path, output_working: bool = False
    ):
        LOG.debug("Controling matrix - %s to - %s", self.adjustment_ufm, self.control_ufm)

        working_dir = out_path / "working"
        if output_working:
            working_dir.mkdir(exist_ok=True)
        LOG.debug("reading in UFMS: adj - %s", self.adjustment_ufm)
        adjustment_matrices: dict[int, pd.DataFrame] = ufm_converter.read_ufm(
            self.adjustment_ufm, saturn_path, self.levels
        )
        LOG.debug("reading in UFMS: cntrl - %s", self.control_ufm)
        control_matrices: dict[int, pd.DataFrame] = ufm_converter.read_ufm(
            self.control_ufm, saturn_path, self.levels
        )

        if adjustment_matrices.keys() != control_matrices.keys():
            raise KeyError("Adjustment matrix and control matrix do not have matching levels.")

        LOG.debug("reading in sector system: %s", self.sector_system_path)
        sector_system = self.sector_system_path.read(
            generic_column_names=True, factors_mandatory=False
        )

        for level, adj_matrix in adjustment_matrices.items():
            LOG.info("Adjusting level: %s", level)

            LOG.debug("labelling and calculating sector matrices")
            adj_matrix_sectors_labelled = _label_sectors_matrix(adj_matrix, sector_system)

            control_matrix = _label_sectors_matrix(control_matrices[level], sector_system)
            if self.exclude_intras:
                adj_matrix_scale = adj_matrix_sectors_labelled.copy()
                adj_matrix_scale.loc[
                    adj_matrix_scale["origin"] == adj_matrix_scale["destination"], "demand"
                ] = 0
                control_matrix.loc[control_matrix["origin"] == control_matrix["destination"], "demand"] = 0
            else:
                adj_matrix_scale = adj_matrix_sectors_labelled.copy()

            adj_sector_matrix = adj_matrix_scale.groupby(
                ["origin_sector", "destination_sector"]
            )["demand"].sum()
            control_sector_matrix = control_matrix.groupby(
                ["origin_sector", "destination_sector"]
            )["demand"].sum()

            LOG.debug("calculating factors")
            sector_factors = control_sector_matrix / adj_sector_matrix
            sector_factors = sector_factors.fillna(1)
            sector_factors.name = "factors"
            ctk.pandas_utils.long_to_wide_infill(sector_factors).to_csv(
                out_path / f"{self.name}_{level}_sector_factors.csv"
            )

            if output_working:
                LOG.debug("outputting intermediary outputs")

                adj_sector_matrix.to_csv(
                    working_dir / f"{self.name}_{level}_var_sector_matrix.csv"
                )
                adj_matrix.to_csv(working_dir / f"{self.name}_{level}_var_matrix.csv")
                control_sector_matrix.to_csv(
                    working_dir / f"{self.name}_{level}_cntrl_sector_matrix.csv"
                )
                control_matrices[level].to_csv(
                    working_dir / f"{self.name}_{level}_cntrl_matrix.csv"
                )

            LOG.debug("adjusting matrix")
            adj_matrix_sectors_labelled = adj_matrix_sectors_labelled.merge(
                sector_factors,
                left_on=["origin_sector", "destination_sector"],
                right_index=True,
            )
            adj_matrix_sectors_labelled["demand"] *= adj_matrix_sectors_labelled["factors"]
            adjusted_matrix = ctk.pandas_utils.long_to_wide_infill(
                adj_matrix_sectors_labelled.set_index(["origin", "destination"])["demand"]
            )
            LOG.debug("writing out matrix")
            adj_matrix_sectors_labelled.set_index(["origin", "destination"])["demand"].to_csv(
                out_path / f"{self.name}_{level}_tuba2_adjusted_matrix.csv", header=False
            )
            adjusted_matrix.to_csv(out_path / f"{self.name}_{level}_sq_adjusted_matrix.csv")

            adj_sector_check = adj_matrix_sectors_labelled.groupby(
                ["origin_sector", "destination_sector"]
            )["demand"].sum()

            if self.exclude_intras:
                sector_system["factors"] = 1
                adj_report = ctk.pandas_utils.MatrixReport(
                    adjusted_matrix,
                    translation_factors=sector_system,
                    translation_from_col="from",
                    translation_to_col="to",
                    translation_factors_col="factors",
                )
                control_report = ctk.pandas_utils.MatrixReport(
                    control_matrices[level],
                    translation_factors=sector_system,
                    translation_from_col="from",
                    translation_to_col="to",
                    translation_factors_col="factors",
                )
                qa_dir = out_path / "QA"
                qa_dir.mkdir(exist_ok=True)
                with pd.ExcelWriter(
                    qa_dir / f"{self.name}_{level}_matrix_report.xlsx"
                ) as writer:
                    ctk.pandas_utils.compare_matrices_and_output(
                        writer, adj_report, control_report, name_a="adj", name_b="control"
                    )
            elif not adj_sector_check.round(CHECK_DP_RESOLUTION).equals(
                control_sector_matrix.round(CHECK_DP_RESOLUTION)
            ):
                
                raise ValueError(
                    "Sector values of the adjusted matrix and control matrix do not match"
                )


def _label_sectors_matrix(
    matrix: pd.DataFrame, sector_translation: pd.DataFrame
) -> pd.DataFrame:
    # convert to long matrix as it makes later joins easier
    long_matrix = ctk.pandas_utils.wide_to_long_infill(matrix, "demand")
    long_matrix.index.names = ["origin", "destination"]
    long_matrix = long_matrix.reset_index()
    long_matrix = long_matrix.merge(sector_translation, left_on="origin", right_on="from")
    long_matrix = long_matrix.merge(
        sector_translation,
        left_on="destination",
        right_on="from",
        suffixes=["_origin", "_destination"],
    )

    return long_matrix.rename(
        columns={"to_origin": "origin_sector", "to_destination": "destination_sector"}
    )[["origin", "origin_sector", "destination", "destination_sector", "demand"]]


class MatrixAdjustmentConfig(ctk.BaseConfig):
    out_path: pathlib.Path
    saturn_path: pathlib.Path
    matrix_scaling_runs: list[MatrixSectorScaling]
    output_working: bool = False

    def run(self) -> None:
        """Run the UFM comparison functionaliy."""
        self.out_path.mkdir(parents=True, exist_ok=True)
        for run in self.matrix_scaling_runs:
            LOG.info("Matrix scaling begining for %s.", run.name)
            run.run(self.saturn_path, self.out_path, self.output_working)

    def log_file_path(self) -> pathlib.Path:
        """Path to the log file for the comparison."""
        self.out_path.mkdir(parents=True, exist_ok=True)
        return self.out_path / "matrix_adjustment.log"
