# copied from 
# https://github.com/fractal-analytics-platform/fractal-helper-tasks/blob/main/tests/conftest.py

import os
import shutil
from pathlib import Path

import pooch
import pytest


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_zarr(testdata_path: Path) -> list[str]:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.10257149"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    platenames = ["plate.zarr", "plate_mip.zarr"]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip": None,
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    for ind, file_name in enumerate(
        [
            "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
        ]
    ):
        # 1) Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        folder = folders[ind]

        # 2) Copy the downloaded Zarr into tests/data
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(Path(zarr_full_path) / file_name, folder)
    return [str(f) for f in folders]


@pytest.fixture(scope="function")
def tmp_zenodo_zarr(zenodo_zarr: list[str], tmpdir: Path) -> list[str]:
    """Generates a copy of the zenodo zarrs in a tmpdir"""
    zenodo_mip_path = str(tmpdir / Path(zenodo_zarr[1]).name)
    zenodo_path = str(tmpdir / Path(zenodo_zarr[0]).name)
    shutil.copytree(zenodo_zarr[0], zenodo_path)
    shutil.copytree(zenodo_zarr[1], zenodo_mip_path)
    return [zenodo_path, zenodo_mip_path]