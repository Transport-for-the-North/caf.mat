# -*- coding: utf-8 -*-
"""Main module for caf.mat package."""

##### IMPORTS #####

import logging


##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


class MatrixError(Exception):
    """Generic matrix error from caf.mat."""
