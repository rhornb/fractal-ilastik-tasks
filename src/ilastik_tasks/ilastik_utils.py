# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Helper functions for ilastik tasks.
Modified from 
https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/tasks/cellpose_utils.py
"""
import logging
from typing import Optional

from pydantic import BaseModel
from pydantic import model_validator
from typing_extensions import Self

from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import ChannelNotFoundError
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import OmeroChannel


logger = logging.getLogger(__name__)


class IlastikChannel1InputModel(ChannelInputModel):
    """
    Channel input for ilastik.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios for
            Cellpose models
    """

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {str(e)}"
            )
            return None


class IlastikChannel2InputModel(BaseModel):
    """
    Channel input for secondary ilastik channel.

    The secondary channel is Optional, thus both wavelength_id and label are
    optional to be set. The `is_set` function shows whether either value was
    set.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios for
            Cellpose models
    """

    wavelength_id: Optional[str] = None
    label: Optional[str] = None

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """
        Check that only 1 of `label` or `wavelength_id` is set.
        """
        wavelength_id = self.wavelength_id
        label = self.label
        if (wavelength_id is not None) and (label is not None):
            raise ValueError(
                "`wavelength_id` and `label` cannot be both set "
                f"(given {wavelength_id=} and {label=})."
            )
        return self

    def is_set(self):
        if self.wavelength_id or self.label:
            return True
        return False

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Second channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {str(e)}"
            )
            return None
