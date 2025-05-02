# -*- coding: utf-8 -*-
"""Tests for `pa_to_od.od_to_pa` module."""

##### IMPORTS #####

# Built-Ins
import logging

# Third Party
import caf.base as base

# Local Imports
from caf.mat import matrices
from caf.mat.pa_to_od import od_to_pa

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class TestDecompileOD:

    def test_integration(self):
        segments = ["m", "tp"]
        from_segmentation = base.Segmentation(
            base.SegmentationInput(
                enum_segments=segments + ["userclass"],
                naming_order=["userclass"] + segments,
                subsets={"m": [3]},
            )
        )
        to_segmentation = base.Segmentation(
            base.SegmentationInput(
                enum_segments=segments + ["p", "direction_od"],
                naming_order=["direction_od", "p"] + segments,
                subsets={"m": [3]},
            )
        )
        zoning = base.ZoningSystem.get_zoning("normits")

        compiled_matrices = matrices.MockMatrices(
            from_segmentation, zoning, type_=matrices.MatrixType.OD
        )
        decompiled_matrices = matrices.MockMatrices(
            to_segmentation, zoning, type_=matrices.MatrixType.OD
        )

        od_to_pa.decompile_od(
            compiled_matrices,
            decompiled_matrices,
            base.segments.SegmentsSuper.USERCLASS.get_segment(),
            base.segments.SegmentsSuper.PURPOSE.get_segment(),
        )
        assert False, "WIP test"
