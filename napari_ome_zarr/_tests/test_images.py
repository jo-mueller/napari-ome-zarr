import pytest

from napari_ome_zarr._tests.conftest import count_layers_in_scene

images = [
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr", # noqa: E501
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0079A/idr0079_images.zarr",  # bf2raw # noqa: E501
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0044A/4007801.zarr", # noqa: E501
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr",  # noqa: E501
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpD_chicken_embryo_MIP.ome.zarr",  # noqa: E501
]

scenes = [
    "https://radosgw.public.os.wwu.de/rfc5-transform-test-data/cells3d_scene_2d_to_3d_tilt.zarr" # noqa: E501
]

plates = [
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0090/190129.zarr" # noqa: E501
    ]


@pytest.mark.parametrize("url", images)
def test_images(url, make_napari_viewer):
    viewer = make_napari_viewer()

    viewer.open(url, plugin="napari-ome-zarr")

    assert len(viewer.layers) > 0


@pytest.mark.parametrize("url", scenes)
def test_scenes(url, make_napari_viewer):
    from napari.layers import Image, Labels
    from ome_zarr import OMEZarrScene

    # open in viewer
    viewer = make_napari_viewer()
    viewer.open(url, plugin="napari-ome-zarr")
    assert len(viewer.layers) > 0

    # count layers in the scene
    # and compare against napari
    scene = OMEZarrScene.from_ome_zarr(url)
    counted = count_layers_in_scene(scene)
    n_images = len([layer for layer in viewer.layers if isinstance(layer, Image)])
    n_labels = len([layer for layer in viewer.layers if isinstance(layer, Labels)])

    assert counted["image_layers"] == n_images
    assert counted["label_layers"] == n_labels


@pytest.mark.parametrize("url", plates)
def test_plates(url, make_napari_viewer):
    viewer = make_napari_viewer()

    viewer.open(url, plugin="napari-ome-zarr")

    assert len(viewer.layers) > 0
