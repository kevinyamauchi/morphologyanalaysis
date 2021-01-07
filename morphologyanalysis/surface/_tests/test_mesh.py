import numpy as np

from morphologyanalysis.surface.mesh import (
    mesh_from_points,
    mesh_from_interface,
)


def test_mesh_from_points():
    points = np.array([[0, 0, 0], [0, 10, 0], [0, 10, 10], [5, 5, 5]])
    mesh = mesh_from_points(points)
    np.testing.assert_equal(mesh.points, points)


def test_mesh_from_interface():
    interface_location = 5
    label_im = np.ones((9, 9, 9))
    label_im[interface_location:, :, :] = 2

    mesh = mesh_from_interface(
        label_im=label_im,
        parent_label=2,
        neighbor_label=1,
        neighbor_distance=1,
    )

    expected_n_points = label_im.shape[1] * label_im.shape[2]
    assert mesh.n_points == expected_n_points
