import numpy as np
from pyvista import PolyData

from .utils import find_interface


def mesh_from_points(points: np.ndarray) -> PolyData:
    """Generate a mesh from a cloud of 2d points using delaunay triangulation.

    Parameters
    ----------
    points : np.ndarray
        The (MxN) array of M points in D dimensions from which to generate the mesh.

    Returns
    -------
    surface_mesh : PolyData
        The pyvista PolyData object containing the mesh.
        See the pyvista docs: https://docs.pyvista.org/core/points.html
    """
    point_cloud = PolyData(points)
    surface_mesh = point_cloud.delaunay_2d()

    return surface_mesh


def mesh_from_interface(
    label_im: np.ndarray,
    parent_label: int,
    neighbor_label: int,
    neighbor_distance: float = 1,
) -> PolyData:
    """Generate a mesh (delaunay triangulation) from a specified interface in a label image

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
    surface_mesh : PolyData
        The pyvista PolyData object containing the mesh.
        See the pyvista docs: https://docs.pyvista.org/core/points.html
    """
    interface_coords = find_interface(
        label_im=label_im,
        parent_label=parent_label,
        neighbor_label=neighbor_label,
        neighbor_distance=neighbor_distance,
    )
    surface_mesh = mesh_from_points(interface_coords)

    return surface_mesh
