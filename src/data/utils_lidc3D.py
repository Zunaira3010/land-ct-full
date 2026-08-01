"""
Adapted from the official LAND repository (aolivtous/LAND_3DChestCT),
src/utils/utils_lidc3D.py — Apache License 2.0.
Copyright 2026 Eurecat, Centre Tecnologic de Catalunya.
See reference_official_repo/LICENSE and reference_official_repo/NOTICE.

CHANGE FROM ORIGINAL (per Apache 2.0 sec. 4(b)): the original file also
defines a torch Dataset class and helpers that import our not-yet-built
vae/unet/pipeline modules (Phase 4+ of this project). Importing that file
as-is would fail here since those modules don't exist yet. This file
extracts only the three functions preprocessing.py actually calls —
`fit_nparray_to_given_size`, `regularize_components_minimal`, and
`extract_3d_contours` — copied verbatim (logic unchanged) so Phase 2
doesn't have a false dependency on Phase 4+ code.
"""
import numpy as np
from scipy.ndimage import find_objects, label


def fit_nparray_to_given_size(arr, size=256):
    h, w, d = arr.shape
    if w != size or h != size or d != size:
        arr_ = np.zeros((size, size, size))
        arr_[:h, :w, :d] = arr[:size, :size, :size]
    else:
        arr_ = arr
    return arr_


def regularize_components_minimal(binary_mask):
    """
    Converts each connected component in a 3D binary mask into the smallest
    odd-sized cube that contains it, and returns a new binary mask with those cubes.

    Parameters:
        binary_mask (np.ndarray): 3D binary mask.

    Returns:
        np.ndarray: New binary mask with regular cubes containing each component.
    """
    assert binary_mask.ndim == 3, "Input mask must be 3D"

    output_mask = np.zeros_like(binary_mask)
    labeled_mask, num_features = label(binary_mask, structure=np.ones((3, 3, 3)))

    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        bbox = find_objects(component)[0]  # tuple of slices per axis

        # Compute the bounding box extents
        min_coords = [s.start for s in bbox]
        max_coords = [s.stop for s in bbox]
        sizes = [stop - start for start, stop in zip(min_coords, max_coords)]

        # Determine minimal odd-sized cube side length N
        max_extent = max(sizes)
        N = max_extent if max_extent % 2 == 1 else max_extent + 1
        half_N = N // 2

        # Compute cube center
        center = [(start + stop) // 2 for start, stop in zip(min_coords, max_coords)]

        # Compute cube bounds, clipped to volume
        bounds = []
        for c, dim in zip(center, binary_mask.shape):
            start = max(0, c - half_N)
            end = min(dim, c + half_N + 1)
            # Adjust bounds to make cube size exactly N if clipped
            actual_size = end - start
            if actual_size < N:
                if start == 0:
                    end = min(dim, N)
                elif end == dim:
                    start = max(0, dim - N)
            bounds.append((start, end))

        z0, z1 = bounds[0]
        y0, y1 = bounds[1]
        x0, x1 = bounds[2]

        output_mask[z0:z1, y0:y1, x0:x1] = 1

    return output_mask


def extract_3d_contours(binary_mask, width=1):
    """
    Extracts the outer contour voxels of 3D connected components with specified thickness.

    Parameters:
        binary_mask (np.ndarray): 3D binary mask.
        width (int): Contour width in number of voxels (>=1).

    Returns:
        np.ndarray: Binary mask with contour voxels of specified width.
    """
    assert binary_mask.ndim == 3, "Input must be a 3D binary mask"
    assert width >= 1, "Contour width must be >= 1"

    current_mask = binary_mask.astype(bool)
    inner_mask = current_mask.copy()

    for _ in range(width):
        padded = np.pad(inner_mask, 1, mode='constant', constant_values=0).astype(bool)

        # Extract 6-connected neighbors
        xm = padded[1:-1, 1:-1, :-2]
        xp = padded[1:-1, 1:-1, 2:]
        ym = padded[1:-1, :-2, 1:-1]
        yp = padded[1:-1, 2:, 1:-1]
        zm = padded[:-2, 1:-1, 1:-1]
        zp = padded[2:, 1:-1, 1:-1]

        # Keep only voxels fully surrounded by neighbors
        surrounded = xm & xp & ym & yp & zm & zp
        inner_mask = inner_mask & surrounded

    contour_mask = current_mask & (~inner_mask)
    return contour_mask.astype(np.uint8)
