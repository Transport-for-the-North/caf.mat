# -*- coding: utf-8 -*-
"""
module_description
"""

##### IMPORTS #####

# Built-Ins
import logging

# Third Party
import caf.base as base
import pandas as pd
from caf.base import segments

# Local Imports
from caf.mat.pa_to_od import matrices

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####
def dummy_scratch():
    from_segmentation = base.Segmentation(
        base.SegmentationInput(
            enum_segments=[segments.SegmentsSuper.USERCLASS, segments.SegmentsSuper.MODE],
            naming_order=["userclass", "m"],
        )
    )

    _get_segmentation_translation(
        from_segmentation,
        segments.SegmentsSuper.USERCLASS.get_segment(),
        segments.SegmentsSuper.PURPOSE.get_segment(),
    )


def _get_segmentation_translation(
    from_segmentation: base.Segmentation, from_seg: segments.Segment, to_seg: segments.Segment
) -> tuple[base.Segmentation, pd.Series]:
    try:
        to_segmentation, lookup = from_segmentation.translate_segment(from_seg, to_seg)
    except FileNotFoundError:
        to_segmentation, lookup = from_segmentation.translate_segment(
            from_seg, to_seg, reverse=True
        )
        lookup = lookup.reset_index().set_index(lookup.name)

    return to_segmentation, lookup


def decompile_od(
    compiled_matrices: matrices.MatricesBase,
    decompiled_matrices: matrices.MatricesBase,
    from_segment: base.Segment,
    to_segment: base.Segment,
):
    if compiled_matrices.type != matrices.MatrixType.OD:
        raise ValueError(f"compiled_matrices should be OD type not {compiled_matrices.type}")

    to_segmentation, lookup = _get_segmentation_translation(
        compiled_matrices.segmentation, from_segment, to_segment
    )

    groupings: dict[int, list[int]] = lookup.groupby(level=0).agg(list).squeeze().to_dict()

    if all(len(i) == 1 for i in groupings.values()):
        # One to one lookup so we don't need decompiled matrices and can just output data as is
        raise NotImplementedError("WIP!")

    # Check if any segment values are found in multiple lists i.e. many-to-many lookup
    unique_to_segs = set()
    for to_segs in groupings.values():
        for i in to_segs:
            if i in unique_to_segs:
                raise ValueError(
                    f"{to_segment.name} segment {i} found in lookup"
                    f" for multiple {from_segment.name} segments"
                )
        unique_to_segs.update(to_segs)

    if not to_segmentation.is_subset(decompiled_matrices.segmentation):
        raise ValueError(
            "Decompiled matrices aren't the correct segmentation should be "
            f"{to_segmentation.names} not {decompiled_matrices.segmentation.names}"
        )

    if compiled_matrices.zoning != decompiled_matrices.zoning:
        raise ValueError(
            f"Zoning systems between compiled ({compiled_matrices.zoning.name}) and"
            f" decompiled ({decompiled_matrices.zoning.name}) matrices are different"
        )

    # Only allows for one slice to be split into multiple slices so we can
    # iterate through compiled segmentation and split each one out
    for from_slice in compiled_matrices.segmentation.iter_slices():
        print(f"{from_slice=}")
        to_segments = groupings.get(from_slice[from_segment.name])
        if to_segments is None:
            raise NotImplementedError("I think this means this slice stays the same?")

        for to_seg in to_segments:
            # Replace from segment value with to segment
            slice_ = from_slice | {to_segment.name: to_seg}
            slice_.pop(from_segment.name)
            print(f"{to_seg=}\n{slice_=}")

            for to_slice in decompiled_matrices.segmentation.iter_slices(slice_):
                print(f"{to_slice=}")

        break
        # Iterate through all slices with given seg

        # Fh / th is another segment so we will need to replace the uc with purpose
        # then disaggregate to the new fh / th segment

    # Get the relevant deccompiled matrices
