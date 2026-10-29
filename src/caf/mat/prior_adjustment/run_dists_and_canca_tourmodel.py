# -*- coding: utf-8 -*-
"""Run the gravity model/adjustment stage followed by CA/NCA disaggregation, from one config."""

# Built-Ins
import logging
import pathlib

# Local Imports
from caf.mat.dia import distribute as dia_distribute
from caf.mat.prior_adjustment import distribute_tourmodel

LOG = logging.getLogger(__name__)


def run_pipeline(cfg: distribute_tourmodel.DistributeConf) -> None:
    """Run stage 1 (gravity model distribution + adjustment) then stage 2 (CA/NCA disaggregation)."""
    LOG.info("Starting stage 1: gravity model distribution + adjustment")
    distribute_tourmodel.main(cfg)
    LOG.info("Finished stage 1")

    LOG.info("Starting stage 2: CA/NCA disaggregation")
    dia_distribute.main(cfg)
    LOG.info("Finished stage 2")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = distribute_tourmodel.DistributeConf.load_yaml(
        pathlib.Path(__file__).parent / "distribute_tourmodel_config.yml"
    )
    run_pipeline(config)
