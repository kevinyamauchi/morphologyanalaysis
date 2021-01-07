import numpy as np
from scipy.spatial import cKDTree


def find_interface(
    label_im: np.ndarray,
    parent_label: int,
    neighbor_label: int,
    neighbor_distance: float = 1,
) -> np.ndarray:
    """Get the coordinates to all pixels in an interface from a parent label to a neighboring label

    Parameters
    ----------
    label_im : np.ndarray
        The label image to extract the interface from
    parent_label : int
        The label value of the object to extract the interface from
    neighbor_label : int
        The label value of the neighbor to the parent the interface should be along
    neighbor_distance : float
        The distance threshold for calling a neighboring pixel.
        The default value is 1 (i.e., touching on the face).

    Returns
    -------
    interface_coords : np.ndarray
        The (MxD) array of coordinates to the pixels belonging to the specified interface.
        D is the number of dimensions in the provided label_im.
    """
    # get the coordinates
    parent_coords = np.argwhere(label_im == parent_label)
    neighbor_coords = np.argwhere(label_im == neighbor_label)

    # create the kdtree and find neighboring points
    kdt = cKDTree(neighbor_coords)
    d, i = kdt.query(parent_coords)

    interface_coords = parent_coords[d <= neighbor_distance]

    return interface_coords
