import numpy as np
import pytest
from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale, OMEZarrScene
from ome_zarr_models.v06.coordinate_transforms import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Sequence,
    Translation,
)
from skimage import data

from napari_ome_zarr._tests.conftest import count_layers_in_scene


def create_overlap_tiles_scene() -> OMEZarrScene:
    """
    Create a scene with four overlapping tiles of a 2D image.
    """
    image = data.human_mitosis()

    world_cs = CoordinateSystem(
        name="world",
        axes=(
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ),
    )

    # split into four overlapping tiles
    tiles = []
    transforms = []
    overlap = 50  # pixels
    height, width = image.shape[:2]
    for y in range(0, height, height // 2):
        for x in range(0, width, width // 2):
            y_end = min(y + height // 2 + overlap, height)
            x_end = min(x + width // 2 + overlap, width)
            tile = image[y:y_end, x:x_end]

            binary = tile > tile.mean()

            oz_binary = OMEZarrImage(
                data=binary,
                axes="yx",
                scale={"y": 0.5, "x": 0.5},
                axes_units={"y": "micrometer", "x": "micrometer"},
                name=f"binary_tile_{y}_{x}",
            )

            oz_binary_ms = OMEZarrLabels(
                oz_binary,
            )

            oz_image = OMEZarrImage(
                data=tile,
                axes="yx",
                scale={"y": 0.5, "x": 0.5},
                axes_units={"y": "micrometer", "x": "micrometer"},
                name=f"tile_{y}_{x}",
            )
            oz_ms = OMEZarrMultiscale(
                oz_image,
                channel_names=["Brightfield"],
                channel_colors=["FFFFFF"],
                labels=oz_binary_ms,
            )

            translation = Translation(
                translation=(y, x),
                input=CoordinateSystemIdentifier(name="physical", path=f"tile_{y}_{x}"),
                output=CoordinateSystemIdentifier(name="world"),
            )
            transforms.append(translation)
            tiles.append(oz_ms)

    return OMEZarrScene(
        images=tiles,
        coordinate_systems=(world_cs,),
        coordinate_transformations=transforms,
    )


def create_YX_to_CZYX_scene() -> OMEZarrScene:
    """
    Create a scene with a 2D image embedded in a 3D coordinate system
    that also has a channel dimension.
    """

    img = data.cells3d().transpose((1, 0, 2, 3))
    some_slice = img[0, 30, :, :]

    oz_img = OMEZarrImage(
        data=img,
        axes=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        name="cells3d",
    )

    oz_ms = OMEZarrMultiscale(
        image=oz_img,
    )

    slice_img = OMEZarrImage(
        data=some_slice, axes=["y", "x"], scale={"y": 1, "x": 1}, name="cells3d_slice"
    )

    slice_ms = OMEZarrMultiscale(
        image=slice_img,
    )

    transform_to_3d = Sequence.model_validate(
        {
            "type": "sequence",
            "input": {"path": "cells3d_slice", "name": "physical"},
            "output": {"path": "cells3d", "name": "physical"},
            "transformations": [
                {"type": "projectAxis", "createdOutputs": [0, 1]},
                {"type": "translation", "translation": [0, 30, 0, 0]},
            ],
        }
    )

    return OMEZarrScene(
        images=[oz_ms, slice_ms], coordinate_transformations=[transform_to_3d]
    )


def create_YXto_ZYX_scene() -> OMEZarrScene:
    """
    Create a scene with a 2D image embedded in a 3D coordinate system.
    """

    img = data.cells3d().transpose((1, 0, 2, 3))
    some_slice = img[0, 30, :, :]

    oz_img = OMEZarrImage(
        data=img[0],
        axes=["z", "y", "x"],
        scale={"z": 1, "y": 1, "x": 1},
        name="cells3d",
    )

    oz_ms = OMEZarrMultiscale(
        image=oz_img,
    )

    slice_img = OMEZarrImage(
        data=some_slice, axes=["y", "x"], scale={"y": 1, "x": 1}, name="cells3d_slice"
    )

    slice_ms = OMEZarrMultiscale(
        image=slice_img,
    )

    transform_to_3d = Sequence.model_validate(
        {
            "type": "sequence",
            "input": {"path": "cells3d_slice", "name": "physical"},
            "output": {"path": "cells3d", "name": "physical"},
            "transformations": [
                {"type": "projectAxis", "createdOutputs": [0]},
                {"type": "translation", "translation": [30, 0, 0]},
            ],
        }
    )

    return OMEZarrScene(
        images=[oz_ms, slice_ms], coordinate_transformations=[transform_to_3d]
    )


@pytest.mark.parametrize(
    "scene",
    [
        create_overlap_tiles_scene(),
        create_YX_to_CZYX_scene(),
        create_YXto_ZYX_scene(),
    ],
)
def test_scene_in_napari(scene, tmp_path, make_napari_viewer):
    from napari.layers import Image, Labels

    scene.to_ome_zarr(str(tmp_path / "tmp_scene.ome.zarr"), overwrite=True)

    viewer = make_napari_viewer()
    viewer.open(path=str(tmp_path / "tmp_scene.ome.zarr"), plugin="napari-ome-zarr")

    # Check every channel and layer is accounted for
    n_layers = count_layers_in_scene(scene)
    assert len(viewer.layers) == n_layers["image_layers"] + n_layers["label_layers"]

    # Make sure we have the correct amount of labels and image layers
    n_labels_layer_viewer = 0
    n_image_layers_viewer = 0
    for layer in viewer.layers:
        if isinstance(layer, Image):
            n_image_layers_viewer += 1
        elif isinstance(layer, Labels):
            n_labels_layer_viewer += 1

    assert n_image_layers_viewer == n_layers["image_layers"]
    assert n_labels_layer_viewer == n_layers["label_layers"]


def test_properties_forwarding(tmp_path, make_napari_viewer):
    """
    This checks whether the layer properties units, labels, etc
    are correctly populated by the napari-ome-zarr plugin.
    """
    scene = create_overlap_tiles_scene()
    scene.to_ome_zarr(str(tmp_path / "tmp_scene.ome.zarr"), overwrite=True)

    viewer = make_napari_viewer()
    viewer.open(path=str(tmp_path / "tmp_scene.ome.zarr"), plugin="napari-ome-zarr")

    # Make sure we populated the properties correctly
    for layer in viewer.layers:
        assert layer.units[0] == "micrometer"
        assert layer.units[1] == "micrometer"
        assert layer.axis_labels == ("y", "x")

    # check that all layers are named appropriately
    for _, ms_image in scene.images.items():
        if hasattr(ms_image, "omero") and ms_image.omero is not None:
            for ch in ms_image.omero.channels:
                ch_name = f"{ms_image.name}: {ch.label}"
                assert ch_name in viewer.layers
        else:
            assert ms_image.name in viewer.layers

        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            for label_name in ms_image.labels.keys():
                assert label_name in viewer.layers

    # check that scale values have been correctly forwarded
    # to image AND labels layers
    for _, ms_image in scene.images.items():
        layers = [
            layer for layer in viewer.layers if layer.name.startswith(ms_image.name)
        ]
        for layer in layers:
            assert np.array_equal(
                layer.scale, np.asarray(list(ms_image.images[0].scale.values()))
            )

        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            for label_name, label_img in ms_image.labels.items():
                layer = viewer.layers[label_name]
                assert np.array_equal(
                    layer.scale, np.asarray(list(label_img.images[0].scale.values()))
                )

    # Check that the affine matrix has been properly set in napari layers
    for _, ms_image in scene.images.items():
        transform = scene._graph.get_sequence(
            (f"{ms_image.name}", "physical"), ("", "world")
        )
        affine = transform.simplify().to_affine().matrix

        layers = [
            layer for layer in viewer.layers if layer.name.startswith(ms_image.name)
        ]
        for layer in layers:
            assert np.array_equal(layer.affine.affine_matrix, affine)

        # If no tranform between image and labels space is specified
        # the affine should propagate to the labels layers as well
        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            for label_name, label_img in ms_image.labels.items():
                layer = viewer.layers[label_name]
                assert np.array_equal(layer.affine.affine_matrix, affine)
