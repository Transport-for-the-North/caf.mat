# -*- coding: utf-8 -*-
"""Time period and from / to home factors required for PA and OD conversions."""

##### IMPORTS #####

# Built-Ins
import abc
import logging
from typing import Iterator

# Local Imports
from caf.mat.pa_to_od import matrices

from caf.base import segments

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class TimePeriod(abc.ABC):

    def get(self, segment: matrices.Segment) -> list[matrices.Matrix]:
        """Get matrix of TP factors for given segment."""


class FromToHome(abc.ABC):

    def get_from(self, segment: matrices.Segment) -> matrices.Matrix:
        """Get matrix of from home factors for each output time period."""

    def get_to(self, segment: matrices.Segment) -> matrices.Matrix: ...


class MissingTP(abc.ABC):
    """Scaling factors for handling missing time periods during combining."""
