import shutil
from pathlib import Path

import pytest
from devtools import debug
from fractal_tasks_core.channels import ChannelInputModel

from ilastik_tasks.ilastik_pixel_classification_segmentation import (
    ilastik_pixel_classification_segmentation,
)

def test_ilastik_pixel_classification_segmentation_task_2D_single_channel(tmp_zenodo_zarr: list[str]):
    """
    Test the 2D (e.g. MIP) ilastik_pixel_classification_segmentation task with single channel input.
    """
    ilastik_model = (Path(__file__).parent / "data/pixel_classifier_2D.ilp").as_posix()
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    
    ilastik_pixel_classification_segmentation(
        zarr_url=zarr_url,
        level=0,
        channel=ChannelInputModel(label="DAPI"),
        channel2=None,
        ilastik_model=str(ilastik_model),
        output_label_name="test_label",
    )

# TODO: add (2D) dual channel ilastik model for testing
# TODO: add (3D) test data for dual channel testing - code works but could not yet find suitable test data