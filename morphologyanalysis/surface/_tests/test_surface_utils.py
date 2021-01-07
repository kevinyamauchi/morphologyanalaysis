import numpy as np

from morphologyanalysis.surface.utils import find_interface


def test_find_interface():
    interface_location = 5
    label_im = np.ones((9, 9, 9))
    label_im[interface_location:, :, :] = 2

    interface_points = find_interface(
        label_im=label_im,
        parent_label=2,
        neighbor_label=1,
        neighbor_distance=1,
    )

    expected_n_points = label_im.shape[1] * label_im.shape[2]
    axis_0 = interface_location * np.ones((expected_n_points,))
    axis_1 = np.repeat(np.arange(label_im.shape[1]), label_im.shape[2])
    axis_2 = np.tile(np.arange(label_im.shape[2]), label_im.shape[1])
    expected_points = list(np.column_stack([axis_0, axis_1, axis_2]))

    np.testing.assert_array_equal(interface_points, expected_points)
