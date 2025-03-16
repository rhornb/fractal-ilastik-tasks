import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug

from fractal_tasks_core.channels import ChannelInputModel

from ilastik_tasks.ilastik_pixel_classification_segmentation import (
    ilastik_pixel_classification_segmentation,
)

# TODO: add 2D testdata

@pytest.fixture(scope="function")
def test_data_dir_3d(tmp_path: Path, zenodo_zarr_3d: list) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "ilastik_data_3d").as_posix()
    debug(zenodo_zarr_3d, dest_dir)
    shutil.copytree(zenodo_zarr_3d, dest_dir)
    return dest_dir


def test_ilastik_pixel_classification_segmentation_task_3D(test_data_dir_3d):
    """
    Test the 3D ilastik_pixel_classification_segmentation task with dual channel input.
    """
    ilastik_model = (Path(__file__).parent / "data/pixel_classifier_3D_dual_channel.ilp").as_posix()
    zarr_url = f"{test_data_dir_3d}/B/03/0"
    
    ilastik_pixel_classification_segmentation(
        zarr_url=zarr_url,
        level=0,
        channel=ChannelInputModel(label="DAPI_2"),
        channel2=ChannelInputModel(label="ECadherin_2"),
        ilastik_model=str(ilastik_model),
        output_label_name="test_label",
    )
    
    # Test failing of task if model was trained with two channels 
    # but only one is provided
    with pytest.raises(ValueError):
            ilastik_pixel_classification_segmentation(
            zarr_url=zarr_url,
            level=0,
            channel=ChannelInputModel(label="DAPI_2"),
            channel2=None,
            ilastik_model=str(ilastik_model),
            output_label_name="test_label_single_channel",
        )
