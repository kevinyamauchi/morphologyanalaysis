from typing import Optional

import numpy as np
from skimage.segmentation import find_boundaries


def label_boundaries(
    label_im: np.ndarray, background_label: Optional[int] = None
) -> np.ndarray:
    """ Creates a new label image containing only the boundaries of a label image.

    Parameters
    ----------
    label_im : np.ndarray
        The label image to find the boundaries in
    background_label : Optional[int]
        The label value for the background. If a background label is provided,
        the interfaces in the background are not labeled in the resulting image.
        If set to None, the interfaces in the all regions will be labeled in the
        resulting image. The default value is None.

    Returns
    -------
    label_boundaries : np.ndarray
        A label image with only the boundaries in label_im labeled.
    """

    if background_label is None:
        boundary_mode = 'thick'
    else:
        boundary_mode = 'inner'
    boundaries = find_boundaries(
        label_im, mode=boundary_mode, background=background_label
    )

    label_boundaries = label_im.copy()
    label_boundaries[np.logical_not(boundaries)] = 0

    return label_boundaries
