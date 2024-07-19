import numpy as np
import cv2
import shapely


def masks2polypoints(masks, strategy="all"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
        masks (numpy array): the output of the model
        strategy (str): 'all', 'concat' or 'largest'. Defaults to all

    Returns:
        polygons (List): list of polygons from masks
    """
    polys = []
    for x in masks.astype("uint8"):
        contours = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for k in contours:
            k = k.reshape(-1, 2).astype("float32")
            polys.append(k)
    return polys


def pix2pc(pixels, width, height):
    """Convert pixel coordinates to pc locations."""
    pc = np.apply_along_axis(
        lambda x: np.r_[x[0] / 1800 * 100, x[1] / 1200 * 100], -1, pixels
    )
    return pc


def simplify_polygon(polygon, tolerance=2.0):
    """Simplify a polygon with shapely.

    Args:
        polygon: ndarray of the polygon positions of N points with the shape (N,2)
        tolerance: float the tolerance (pixels)

    Returns:
        ndarray
    """
    try:
        poly = shapely.geometry.Polygon(polygon)
        poly_s = poly.simplify(tolerance=tolerance)
    except ValueError:
        # simplification not possible
        return polygon
    # convert it back to numpy
    return np.array(poly_s.boundary.coords[:])
