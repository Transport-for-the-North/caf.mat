# -*- coding: utf-8 -*-
"""Compare OD to PA outputs from CAF.mat to original process."""

##### IMPORTS #####

# Built-Ins
import logging
import pathlib
import warnings
from collections.abc import Iterable

# Third Party
import caf.toolkit as ctk
import numpy as np
import pandas as pd
import pydantic
from pydantic import dataclasses

# Local Imports
import caf.mat

##### CONSTANTS #####

_NAME = pathlib.Path(__file__).stem
LOG = logging.getLogger(_NAME)
CONFIG_PATH = pathlib.Path(__file__).with_name("od2pa_comparison.yml")


##### CLASSES & FUNCTIONS #####


class MatrixComparison:

    def __init__(self, old: pd.DataFrame, new: pd.DataFrame):
        self._old = old
        self._new = new
        self._difference = None
        self._percentage = None

    @property
    def difference(self) -> pd.DataFrame:
        if self._difference is None:
            self._difference = self._new - self._old
        return self._difference.copy()

    @property
    def percentage_difference(self) -> pd.DataFrame:
        if self._percentage is None:
            old = self._old.to_numpy()
            new = self._new.to_numpy()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
                diff = np.divide(new, old, where=~((new == 0) & (old == 0))) - 1
            self._percentage = pd.DataFrame(
                diff, index=self._new.index, columns=self._new.columns
            )

        return self._percentage

    def _summarise(self, data: np.ndarray, name: str) -> dict[str, float]:
        size = np.size(data)
        finite = np.sum(np.isfinite(data))
        return {
            f"{name}_nans": np.sum(np.isnan(data)),
            f"{name}_infs": np.sum(np.isinf(data)),
            f"{name}_cells": size,
            f"{name}_finite": finite,
            f"{name}_finite_proportion": finite / size,
            f"{name}_zeros": np.sum(data == 0),
            f"{name}_negatives": np.sum(data < 0),
            f"{name}_total": np.sum(data),
        }

    def compare(self) -> dict[str, float]:
        old = self._old.to_numpy()
        new = self._new.to_numpy()
        percentage = self.percentage_difference.to_numpy()
        diff = self.difference.to_numpy()

        comparison = {
            **self._summarise(new, "new"),
            **self._summarise(old, "old"),
        }
        comparison.update(
            {
                "total_difference": comparison["new_total"] - comparison["old_total"],
                "total_percentage_difference": (
                    comparison["new_total"] / comparison["old_total"]
                )
                - 1,
                "max_abs_difference": np.max(np.abs(diff)),
                "abs_difference_nans": np.sum(np.isnan(diff)),
                "max_percentage_difference": np.max(
                    np.abs(percentage), where=np.isfinite(percentage), initial=0
                ),
                "percentage_difference_nans": np.sum(np.isnan(percentage)),
                "percentage_difference_infs": np.sum(np.isinf(percentage)),
            }
        )
        return comparison

    def comparison_summary(self) -> str:
        template = (
            "\tOld Total: {old_total:.1e}\n\tNew Total: {new_total:.1e}"
            "\n\tTotal Difference: {total_difference:.1e} ({total_percentage_difference:.1%})"
            "\n\tMax Abs Difference: {max_abs_difference:.1e}"
            " ({abs_difference_nans:,.0f} NaNs)"
            "\n\tMax %% Difference: {max_percentage_difference:.1e} ({percentage_difference_nans:,.0f} Nans"
            " {percentage_difference_infs:,.0f} Infs)"
        )

        comparison = self.compare()
        return template.format_map(comparison)


def compare_matrices(
    matrices: Iterable[tuple[pathlib.Path, pathlib.Path]], output_folder: pathlib.Path
):
    summary_data = {}
    for old_path, new_path in matrices:
        LOG.info("Loading matrices %s, %s", old_path.name, new_path.name)
        try:
            old = ctk.io.read_csv_matrix(old_path, format_="square")
            new = ctk.io.read_csv_matrix(new_path, format_="square")
        except FileNotFoundError as exc:
            LOG.error("File doesn't exist: %s", exc)
            continue

        if not old.index.equals(new.index):
            raise ValueError("new and old indices aren't the same")

        if old.equals(new):
            LOG.info("Matrices equal for: %s, %s", old_path.name, new_path.name)
            summary_data[new_path.stem] = {"equal": True}

        comparison = MatrixComparison(old, new)
        comparison.difference.to_csv(output_folder / (new_path.stem + "-diff.csv"))
        comparison.percentage_difference.to_csv(
            output_folder / (new_path.stem + "-perc_diff.csv")
        )

        LOG.error(
            "Matrices are different for: %s, %s\n%s",
            old_path.name,
            new_path.name,
            comparison.comparison_summary(),
        )

        summary_data[new_path.stem] = comparison.compare()

    out_path = output_folder / "comparison_summary.csv"
    summary = pd.DataFrame.from_dict(summary_data, orient="index")
    summary.to_csv(out_path)


def compare_pa_outputs(
    old_folder: pathlib.Path, new_folder: pathlib.Path, output_folder: pathlib.Path
):
    lookup = [
        ("hb_pa_yr2023_uc1_m3.csv", "PA_m3_business_hb.csv.bz2"),
        ("hb_pa_yr2023_uc2_m3.csv", "PA_m3_commute_hb.csv.bz2"),
        ("hb_pa_yr2023_uc3_m3.csv", "PA_m3_other_hb.csv.bz2"),
        ("nhb_pa_yr2023_uc4_m3.csv", "PA_m3_business_nhb.csv.bz2"),
        ("nhb_pa_yr2023_uc5_m3.csv", "PA_m3_other_nhb.csv.bz2"),
    ]
    LOG.info(
        "Comparing PA outputs matrices:\n\tNew: %s\n\tOld: %s",
        new_folder.resolve(),
        old_folder.resolve(),
    )

    compare_matrices(((old_folder / i, new_folder / j) for i, j in lookup), output_folder)


def compare_disaggregated_od(
    old_folder: pathlib.Path, new_folder: pathlib.Path, output_folder: pathlib.Path
):
    old_name = "noham_m3_ts{ts}_uc{uc}{ft}.csv"  # noham_m3_ts1_uc1fr.csv
    new_name = "OD_m3_ts{ts}_{uc_name}_{ft}.csv.bz2"  # OD_m3_ts1_business_fr.csv.bz2

    paths = []
    for ts in (1, 2, 3, 4):
        for uc, name in ((1, "business"), (2, "commute"), (3, "other")):
            for from_to in ("fr", "to", "nhb"):
                paths.append(
                    (
                        old_folder / old_name.format(ts=ts, uc=uc, ft=from_to),
                        new_folder / new_name.format(ts=ts, uc_name=name, ft=from_to),
                    )
                )

    output_folder = output_folder / "disaggregated od"
    output_folder.mkdir(exist_ok=True)

    LOG.info(
        "Comparing disaggregated OD matrices:\n\tNew: %s\n\tOld: %s",
        new_folder.resolve(),
        old_folder.resolve(),
    )

    compare_matrices(paths, output_folder)


@dataclasses.dataclass
class _Folders:
    old: pydantic.DirectoryPath
    new: pydantic.DirectoryPath


class _Parameters(ctk.BaseConfig):
    final_outputs: _Folders
    disaggregated_od: _Folders
    output_folder: pydantic.DirectoryPath


def main() -> None:
    parameters = _Parameters.load_yaml(CONFIG_PATH)

    details = ctk.ToolDetails("caf.mat.compare", caf.mat.__version__)
    parameters.output_folder.mkdir(exist_ok=True)

    log_file = parameters.output_folder / "compare.log"

    with ctk.LogHelper("", details, log_file=log_file):
        LOG.debug("Parameters:\n%s", parameters.to_yaml())

        compare_disaggregated_od(
            parameters.disaggregated_od.old,
            parameters.disaggregated_od.new,
            parameters.output_folder,
        )
        compare_pa_outputs(
            parameters.final_outputs.old,
            parameters.final_outputs.new,
            parameters.output_folder,
        )


##### MAIN #####
if __name__ == "__main__":
    main()
