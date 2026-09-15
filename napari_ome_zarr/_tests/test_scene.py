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
                scale={"y": 1.0, "x": 1.0},
                axes_units={"y": "micrometer", "x": "micrometer"},
                name=f"binary_tile_{y}_{x}",
            )

            oz_binary_ms = OMEZarrLabels(
                oz_binary,
            )

            oz_image = OMEZarrImage(
                data=tile,
                axes="yx",
                scale={"y": 1.0, "x": 1.0},
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


def _count_layers_in_scene(scene: OMEZarrScene) -> dict:
    """
    Browse the images in a scene and sum up how many layers
    should be created. One channel corresponds to one layer.
    One label image corresponds to one layer.
    """
    n_image_layers: int = 0
    n_label_layers: int = 0
    for _, ms_image in scene.images.items():
        img = ms_image.images[0]

        if "c" in img.axes:
            ch_axis = "".join(img.axes).find("c")
            n_channels = int(img.data.shape[ch_axis])
            n_image_layers += n_channels
        else:
            n_image_layers += 1

        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            n_label_layers += len(ms_image.labels.items())

    return {"image_layers": n_image_layers, "label_layers": n_label_layers}


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
    n_layers = _count_layers_in_scene(scene)
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


def test_units_labels_forwarding(tmp_path, make_napari_viewer):
    """
    This checks whether the layer properties units, labels, etc
    are correctly populated by the napari-ome-zarr plugin.
    """
    scene = create_overlap_tiles_scene()
    scene.to_ome_zarr(str(tmp_path / "tmp_scene.ome.zarr"), overwrite=True)

    viewer = make_napari_viewer()
    viewer.open(path=str(tmp_path / "tmp_scene.ome.zarr"), plugin="napari-ome-zarr")

    for layer in viewer.layers:
        if hasattr(layer, "metadata") and "units" in layer.metadata:
            assert layer.metadata["units"] is not None



if __name__ == "__main__":
    pytest.main([__file__])
