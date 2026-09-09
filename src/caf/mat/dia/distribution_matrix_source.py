# -*- coding: utf-8 -*-
"""Read gravity-model/adjustment stage output matrices directly from their nested output folders."""

# Built-Ins
import pathlib

# Third Party
import caf.base as cb
import pandas as pd

# Local Imports
from caf.mat.matrices import Matrix, MatricesBase, MatrixType
from caf.mat.prior_adjustment.distribute_tourmodel import (
    _ADJ_CHOICE_CONFIG,
    _use_as_list,
    DistributeConf,
)

_ADJ_TARGET_ORDER = ("triple_targ", "o_target", "d_target", "sec_target")


class DistributionMatrixReader(MatricesBase):
    """Read-only matrix source that resolves matrices from distribute_tourmodel.py's
    nested `output_path/p{purpose}/{slice_name}/...` output structure, avoiding the
    need to flatten distribution and adjustment outputs into a single folder for `MatrixFiles`.
    """

    def __init__(
        self,
        cfg: DistributeConf,
        *,
        mode_subset: int | list[int],
        tp_subset: int | list[int],
    ):
        segmentation = cb.Segmentation(
            cb.SegmentationInput(
                enum_segments=cfg.adj_target_matrices["naming_order"],
                naming_order=cfg.adj_target_matrices["naming_order"],
                subsets={
                    "m": _use_as_list(mode_subset),
                    "p": _use_as_list(cfg.purpose_subset),
                    "tp": _use_as_list(tp_subset),
                    "direction_od": _use_as_list(cfg.direction_subset),
                },
            )
        )
        zoning = cb.ZoningSystem.get_zoning(cfg.zone_system)
        super().__init__(segmentation, zoning, MatrixType.OD)
        self._output_path = pathlib.Path(cfg.output_path)
        self._run_adjust = cfg.run_options["run_adjust"]
        choice_key = tuple(
            cfg.adj_target_options[name]["apply"] for name in _ADJ_TARGET_ORDER
        )
        self._adj_folder = _ADJ_CHOICE_CONFIG[choice_key][0] if self._run_adjust else None

    @property
    def name(self) -> str:
        return self._output_path.name

    def _matrix_path(self, slice_: cb.segmentation.SegmentationSlice) -> pathlib.Path:
        purpose = f"p{slice_.get('p')}"
        slice_name = slice_.generate_name()
        slice_dir = self._output_path / purpose / slice_name
        if self._run_adjust:
            return slice_dir / self._adj_folder / f"{slice_name}_matrix.csv"
        return slice_dir / "overall_matrix.csv"

    def _get_matrix(self, slice_: cb.segmentation.SegmentationSlice) -> Matrix:
        path = self._matrix_path(slice_)
        data = pd.read_csv(path, index_col=0)
        data.columns = data.columns.astype(int)
        data.index.name = "origin"
        data.columns.name = "destination"
        return Matrix(data, slice_)

    def _set_matrix(self, matrix: pd.DataFrame, slice_: cb.segmentation.SegmentationSlice) -> None:
        raise PermissionError("DistributionMatrixReader is read-only.")

    def new(self, name: str, **kwargs) -> "DistributionMatrixReader":
        raise NotImplementedError("DistributionMatrixReader does not support creating new matrix sets.")

    def exists(self) -> bool:
        return all(self._matrix_path(slice_).is_file() for slice_ in self.segmentation.iter_slices())
