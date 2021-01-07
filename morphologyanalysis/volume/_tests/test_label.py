import numpy as np

from morphologyanalysis.volume.label import label_boundaries


def create_label_im():
    label_im = np.ones((100, 100), dtype=np.uint)
    label_im[50:, :] = 2

    return label_im


def test_label_boundaries_no_bg_label():
    label_im = create_label_im()
    edge_im = label_boundaries(label_im, background_label=None)

    expected_edge_im = np.zeros_like(label_im)
    expected_edge_im[49, :] = 1
    expected_edge_im[50, :] = 2

    np.testing.assert_equal(edge_im, expected_edge_im)


def test_label_boundaries_bg_label():
    label_im = create_label_im()
    edge_im = label_boundaries(label_im, background_label=1)

    expected_edge_im = np.zeros_like(label_im)
    expected_edge_im[50, :] = 2

    np.testing.assert_equal(edge_im, expected_edge_im)
