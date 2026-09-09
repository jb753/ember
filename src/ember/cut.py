"""Marching cubes algorithm and 3D structured grid cutting utilities.

This module implements the marching cubes algorithm for extracting isosurfaces from 3D
structured grids, along with utilities for creating both structured meridional cuts and
unstructured triangulated cuts. The marching cubes implementation uses a precomputed lookup
table (TRITABLE) to determine how to triangulate the cell edges an isosurface intersects.
Key functionality includes extracting
unstructured triangular cuts using signed distance fields, creating structured 2D meridional
slices by interpolating along grid lines, and converting between structured quad meshes and
unstructured triangle meshes. The module supports both direct marching cubes output and
subsequent interpolation of unstructured data back onto structured grids for easier analysis.

Cuts along grid surfaces by index
---------------------------------

The functions here extract cuts along an arbitrary meridional curve. To take a
cut along a grid surface of constant index, none of them are needed: a
:class:`~ember.block.Block` supports numpy-style indexing, so ``block[i]`` or
``block[:, j]`` returns a lower-dimensional structured cut that shares the
parent's data as a zero-copy view. See the indexing section of
:mod:`ember.block` for the full rules.
"""

import numpy as np
from pykdtree.kdtree import KDTree

import ember.fortran
import ember.grid
from ember import util
from ember.block import Block

_TRITABLE = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
        [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
        [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
        [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
        [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
        [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
        [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
        [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
        [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
        [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
        [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
        [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
        [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
        [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
        [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
        [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
        [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
        [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=int,
)


def _vijk(shape, v):
    """Get (ni-1,nj-1, nk-1) indexing vectors that will extract vertex number v
    from a 3D matrix of size (ni, nj, nk)."""
    ni, nj, nk = shape
    if v == 0:
        return slice(0, ni - 1), slice(0, nj - 1), slice(0, nk - 1)
    elif v == 1:
        return slice(1, ni), slice(0, nj - 1), slice(0, nk - 1)
    elif v == 2:
        return slice(1, ni), slice(1, nj), slice(0, nk - 1)
    elif v == 3:
        return slice(0, ni - 1), slice(1, nj), slice(0, nk - 1)
    elif v == 4:
        return slice(0, ni - 1), slice(0, nj - 1), slice(1, nk)
    elif v == 5:
        return slice(1, ni), slice(0, nj - 1), slice(1, nk)
    elif v == 6:
        return slice(1, ni), slice(1, nj), slice(1, nk)
    elif v == 7:
        return slice(0, ni - 1), slice(1, nj), slice(1, nk)


def _eijk(i, j, k, e):
    """Get 2 (i,j,k) indexing vectors that will extract the start and end
    points of edge e for cell i, j, k of a 3D matrix."""
    if e == 0:
        return (i, j, k), (i + 1, j, k)
    elif e == 1:
        return (i + 1, j, k), (i + 1, j + 1, k)
    elif e == 2:
        return (i + 1, j + 1, k), (i, j + 1, k)
    elif e == 3:
        return (i, j + 1, k), (i, j, k)
    elif e == 4:
        return (i, j, k + 1), (i + 1, j, k + 1)
    elif e == 5:
        return (i + 1, j, k + 1), (i + 1, j + 1, k + 1)
    elif e == 6:
        return (i + 1, j + 1, k + 1), (i, j + 1, k + 1)
    elif e == 7:
        return (i, j + 1, k + 1), (i, j, k + 1)
    elif e == 8:
        return (i, j, k), (i, j, k + 1)
    elif e == 9:
        return (i + 1, j, k), (i + 1, j, k + 1)
    elif e == 10:
        return (i + 1, j + 1, k), (i + 1, j + 1, k + 1)
    elif e == 11:
        return (i, j + 1, k), (i, j + 1, k + 1)


# How many triangles each corner sign pattern makes, and the corner offsets
# of a cell's 8 vertices and of the two ends of its 12 edges. All are derived
# from the table and helpers above rather than written out again, and are
# handed to the Fortran kernels so that they hold no copy of their own.
_NTRITABLE = np.count_nonzero(_TRITABLE != -1, axis=1) // 3
_VERT_IJK = np.array(
    [[s.start for s in _vijk((2, 2, 2), v)] for v in range(8)], dtype=np.int32
)
_EDGE_IJK = np.array([_eijk(0, 0, 0, e) for e in range(12)], dtype=np.int32)


def _cube_index(d):
    """For a 3D array of signed distances, get cube indices."""
    ni, nj, nk = d.shape
    ind = np.zeros((ni - 1, nj - 1, nk - 1), dtype=np.int32, order="F")
    ember.fortran.marching_cubes_index(d, _VERT_IJK, ind)
    return ind


def _marching_cubes(data, dist):
    """Take an unstructured cut of 3D data using a signed distance field.

    Parameters
    ----------
    data: (ni, nj, nk, nvar) array
        Data variables arranged on a three-dimensional structured grid
        following ember convention with components in last axis.
    dist: (ni, nj, nk) array
        The signed distance field at each grid point, `dist=0` sets the cut location.

    Returns
    -------
    triangles: (ntri, 3, nvar) array
        The unstructured cut is composed of `ntri` triangles, each with 3
        vertices, holding values for each of the `nvar` data variables.

    """

    # Which of the 256 corner sign patterns each cell has, and so how many
    # triangles it contributes
    icube = _cube_index(dist)
    ntri_cell = _NTRITABLE[icube]

    # Where each cell's triangles start in the output. Counting in C order
    # walks the cells i first, which is the order the triangles come back in.
    ntri_flat = ntri_cell.ravel(order="C")
    offsets = np.cumsum(ntri_flat) - ntri_flat
    ntri = int(ntri_flat.sum())

    # Nothing is cut
    if ntri == 0:
        return None

    # Only the cells that contribute a triangle reach the kernel, which for a
    # cut of a 3D block is a couple of percent of them
    cells = np.argwhere(ntri_cell > 0)

    ijk_lo = np.zeros((ntri, 3, 3), dtype=np.int32, order="F")
    ijk_hi = np.zeros((ntri, 3, 3), dtype=np.int32, order="F")
    frac = np.zeros((ntri, 3), order="F")

    ember.fortran.marching_cubes_edges(
        dist,
        cells,
        offsets.reshape(ntri_cell.shape)[tuple(cells.T)],
        icube[tuple(cells.T)],
        _TRITABLE,
        _EDGE_IJK,
        ijk_lo,
        ijk_hi,
        frac,
    )

    # Interpolate every variable onto the vertices the kernel located. Done
    # here rather than in the kernel so that the data keeps whatever precision
    # it arrived in, and never has to be handed over in bulk.
    data_lo = data[ijk_lo[..., 0], ijk_lo[..., 1], ijk_lo[..., 2]]
    data_hi = data[ijk_hi[..., 0], ijk_hi[..., 1], ijk_hi[..., 2]]

    return data_lo + (data_hi - data_lo) * frac[..., np.newaxis]


def signed_distance(xr, xr_query):
    """Distance above or below a piecewise line in meridional plane.

    Note that this becomes increasingly inaccurate far away from the
    curve but the zero level is correct (which is sufficient for cutting).

    Parameters
    ----------
    xr : Array, shape (nseg, 2)
        Coordinates of the cut plane segments following ember convention
        with components in last axis.
    xr_query : Array, shape (..., 2)
        Meridional coordinates to evaluate distance at.

    Returns
    -------
    Array, shape (...)
        Signed distance above or below the cut.

    """

    assert xr.shape[-1] == 2, "Segments must have shape (..., 2)"
    assert xr_query.shape[-1] == 2, "Points must have shape (..., 2)"
    assert xr.ndim >= 2, "Segments must be at least 2D"

    shape = xr_query.shape[:-1]

    # A single node spans no segment, so nothing is ever nearer than the
    # sentinel.  The kernel would return its own, so screen the case out here.
    if xr.shape[0] < 2:
        return np.full(shape, np.inf)

    # The kernel indexes points fastest and components slowest, which is the
    # Fortran ordering ember stores nodal arrays in, so the reshape below is a
    # view for a block coordinate array rather than a repack.  Widening to
    # double is a copy, since those arrays are single precision, but it is one
    # pass against the numpy version's ten per segment, and it keeps the return
    # dtype this function has always had.
    xr_query = np.asfortranarray(xr_query, dtype=np.float64)
    d = ember.fortran.signed_distance(
        np.asfortranarray(xr, dtype=np.float64),
        xr_query.reshape(-1, 2, order="F"),
    )

    return d.reshape(shape, order="F")


def _cut_reaches_block(xr_cut, block):
    """Whether a cut curve comes near enough to a block to be worth evaluating.

    Clips each segment of the curve against the block's meridional bounding
    box. A curve that reaches no block's box cannot cross that block, so its
    distance field need never be built.

    The screen is conservative: a segment that enters the bounding box but
    misses the block itself is still accepted, and the sign of the distance
    field then rejects it as before.

    Parameters
    ----------
    xr_cut : array_like, shape (n_segments, 2)
        Meridional :math:`(x, r)` curve segments defining the cut surface.
    block : Block
        Block to test the curve against.

    Returns
    -------
    bool
        False only when no segment of the curve enters the block's box.

    """
    # Taken from the nondimensional coordinates and scaled as four scalars,
    # rather than from block.xrt, which would copy and scale every node
    xrt_nd = block.xrt_nd
    lo = np.array([xrt_nd[..., 0].min(), xrt_nd[..., 1].min()]) * block.L_ref
    hi = np.array([xrt_nd[..., 0].max(), xrt_nd[..., 1].max()]) * block.L_ref

    xr_cut = np.asarray(xr_cut)

    # The same contract signed_distance asserts, checked here because the
    # screen now sees the curve first
    assert xr_cut.shape[-1] == 2, "Segments must have shape (..., 2)"
    assert xr_cut.ndim >= 2, "Segments must be at least 2D"

    seg_start = xr_cut[:-1]
    seg_delta = xr_cut[1:] - seg_start

    # Clip the segment parameter, which runs over [0, 1] along the segment, to
    # the slab between each pair of opposite box faces in turn. What survives
    # in both components at once is the part of the segment inside the box.
    t_in = np.zeros(len(seg_start))
    t_out = np.ones(len(seg_start))
    for c in range(2):
        # A segment constant in this component never crosses either face, so
        # it either lies within the slab for its whole length or misses it
        constant = seg_delta[:, c] == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (lo[c] - seg_start[:, c]) / seg_delta[:, c]
            t_hi = (hi[c] - seg_start[:, c]) / seg_delta[:, c]

        t_in = np.maximum(t_in, np.where(constant, 0.0, np.minimum(t_lo, t_hi)))
        t_out = np.minimum(t_out, np.where(constant, 1.0, np.maximum(t_lo, t_hi)))

        outside = constant & ((seg_start[:, c] < lo[c]) | (seg_start[:, c] > hi[c]))
        t_in = np.where(outside, 1.0, t_in)
        t_out = np.where(outside, 0.0, t_out)

    # An empty interval on every segment means the curve misses the box. A
    # curve of a single node spans no segment and misses by the same token.
    return bool(np.any(t_in <= t_out))


def unstructured(grid, xr_cut):
    r"""Take an unstructured cut through a grid using marching cubes.

    Extracts a triangulated isosurface where the meridional cut curve
    intersects the grid, by running the marching cubes algorithm on the
    signed distance field of each block. Triangles from all intersected
    blocks are concatenated into a single unstructured block.

    Parameters
    ----------
    grid : Grid or Block
        Grid containing blocks to cut, or a single block.
    xr_cut : array_like, shape (n_segments, 2)
        Meridional :math:`(x, r)` curve segments defining the cut surface.

    Returns
    -------
    cut : Block, shape (ntri, 3) or None
        Unstructured triangulated cut, or None if the curve does not
        intersect any block.
    """
    # Convert single Block to list for consistent handling
    if isinstance(grid, Block):
        grid = [grid]

    # Loop over blocks
    triangles = []
    last_block = None
    for block in grid:
        # Reject the blocks the curve does not reach before building any
        # distance field. This also keeps out a block that the curve misses
        # but whose field still changes sign, which happens across the medial
        # axis between two segments, where the nearest segment switches and
        # takes the sign with it without the distance passing through zero.
        if not _cut_reaches_block(xr_cut, block):
            continue

        # Scaled from the nondimensional coordinates rather than read from
        # block.xrt, which also copies the circumferential component
        xr_coords = block.xrt_nd[..., :2] * block.L_ref

        # Evaluate signed distance for all points in the block
        dist = signed_distance(xr_cut, xr_coords)

        # Skip blocks that do not intersect the cut
        if np.all(dist >= 0) or np.all(dist <= 0):
            continue

        # Get triangles for this block
        triangles_block = _marching_cubes(block._data, dist)

        # Add triangles to the list
        if triangles_block is not None:
            triangles.append(triangles_block)
            last_block = block

    # Join all blocks into one array
    if triangles:
        triangles = np.concatenate(triangles)  # Shape (ntri, 3, nvar)

        out = last_block.empty(shape=triangles.shape[:-1])  # Shape (ntri, 3)
        out._data[:] = triangles
        out.set_triangulated(True)

        return out

    # Return None if no triangles found
    return None


def _first_j_crossing(data, dist):
    r"""Interpolate a block's data to the first sign change along ``j``.

    Walks every :math:`(i, k)` grid line, finds the first ``j`` interval over
    which the signed distance changes sign, and interpolates the data linearly
    to the zero. Lines with no sign change, and lines whose bracketing
    distances are too close together to interpolate between, come back as NaN.

    Parameters
    ----------
    data : Array, shape (ni, nj, nk, nvar)
        Nodal block data to interpolate.
    dist : Array, shape (ni, nj, nk)
        Signed distance to the cut curve at every node.

    Returns
    -------
    Array, shape (ni, nk, nvar)
        Data on the cut, NaN on the lines that do not cross it.

    """
    ni, nj, nk = dist.shape

    cut_data = np.full((ni, nk, data.shape[-1]), np.nan)

    # A single j station spans no interval to cross, so no line can cut
    if nj < 2:
        return cut_data

    # Comparing signs rather than differencing them keeps a node sitting
    # exactly on the curve a sign change on both sides of itself, which is
    # what walking the line one interval at a time gives.
    sgn = np.sign(dist)
    changes = sgn[:, 1:, :] != sgn[:, :-1, :]  # Shape (ni, nj-1, nk)

    # First sign change along j. argmax reports the first True, and zero for a
    # line with no sign change at all, which found masks off below.
    found = changes.any(axis=1)  # Shape (ni, nk)
    j_cut = changes.argmax(axis=1)  # Shape (ni, nk)

    # Distances bracketing the cut, gathered from each line's own j_cut
    j_gather = j_cut[:, np.newaxis, :]  # Put back the j axis to gather along
    d1 = np.take_along_axis(dist, j_gather, axis=1)[:, 0, :]
    d2 = np.take_along_axis(dist, j_gather + 1, axis=1)[:, 0, :]

    # A line whose bracketing distances are too close to interpolate between
    # is left uncut, rather than moved on to its next crossing
    delta = d2 - d1
    valid = found & (np.abs(delta) >= 1e-12)

    # Linear interpolation, clamped to the interval it came from.  The invalid
    # lines divide by a zero delta here, and are masked off immediately after.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip(-d1 / delta, 0.0, 1.0)

    # Interpolate every variable across the bracketing pair of j stations
    j_gather = j_gather[..., np.newaxis]  # Broadcast over the variables
    data1 = np.take_along_axis(data, j_gather, axis=1)[:, 0, :, :]
    data2 = np.take_along_axis(data, j_gather + 1, axis=1)[:, 0, :, :]
    cut_data[valid] = (
        data1[valid] + (data2[valid] - data1[valid]) * frac[valid, np.newaxis]
    )

    return cut_data


def structured_meridional(grid, xr_cut):
    r"""Slice a grid along a meridional curve into structured 2D cuts.

    For each block, walks every :math:`(i, k)` grid line and linearly
    interpolates to the first ``j`` location where the signed distance to the
    cut curve changes sign, producing one structured ``(ni, nk)`` cut per
    intersected block.

    Parameters
    ----------
    grid : Grid or Block
        Grid containing blocks to cut, or a single block.
    xr_cut : array_like, shape (n_segments, 2)
        Meridional :math:`(x, r)` curve segments defining the cut surface.

    Returns
    -------
    cut : Grid
        Grid of 2D structured blocks, each shape (ni, nk), holding the cut
        data. Blocks that the curve does not intersect are omitted, so the
        grid is empty when there is no intersection.
    """
    # Convert single Block to list for consistent handling
    if isinstance(grid, Block):
        grid = [grid]

    cut_blocks = []

    for block in grid:
        # Reject the blocks the curve does not reach before building any
        # distance field, as in unstructured() above
        if not _cut_reaches_block(xr_cut, block):
            continue

        # Get meridional coordinates, scaled from the nondimensional ones
        # rather than read from block.xrt, which also copies theta
        xr_coords = block.xrt_nd[..., :2] * block.L_ref

        # Get signed distance
        dist = signed_distance(xr_cut, xr_coords)

        # Check for intersection
        if np.all(dist >= 0) or np.all(dist <= 0):
            continue

        ni, _, nk = block.shape

        # Interpolate to the first j crossing on every (i, k) line
        cut_data = _first_j_crossing(block._data, dist)

        # Check if we found any valid cuts
        if not np.all(np.isnan(cut_data[..., 0])):
            # Create output block
            out_block = block.empty(shape=(ni, nk))
            out_block._data = cut_data
            cut_blocks.append(out_block)

    return ember.grid.Grid(cut_blocks)


def interpolate_to_structured(
    unstructured_block, interp_shape, Beta=0.0, periodic=True, rtol=0.05
):
    r"""Interpolate an unstructured triangular cut onto a structured grid.

    Builds a structured ``(ni, nj)`` grid that **conforms to the meridional
    line** of the cut and **resolves the circumferential direction**, then
    interpolates the triangle-vertex point cloud onto it:

    - index ``i`` runs along the meridional line. A constant-``i`` gridline has
      ``(x, r) = const`` and varies only in theta. Nodes are
      cosine-clustered along the normalised arc length :math:`\zeta \in [0, 1]`
      (double-sided cosine clustering), resolving both extremities of
      the line.
    - index ``j`` runs in theta. A constant-``j`` gridline has
      ``theta = const`` (straight lines), laid out uniformly over one full
      pitch.

    Straight constant-theta lines rely on the solution being periodic in theta
    with period ``pitch = 2*pi/Nb``: source points are wrapped modulo the pitch
    so a single uniform theta window is fully covered however the cut's theta
    band skews along the line. Flow variables are interpolated in the unfolded
    :math:`(\zeta, \theta)` unit space using linear interpolation with a
    nearest-neighbour fallback at the arc-length ends.

    Parameters
    ----------
    unstructured_block : Block, shape (ntri, 3)
        Unstructured triangulated cut, for example from :func:`unstructured`.
    interp_shape : tuple of int
        Target structured grid shape ``(ni, nj)``.
    Beta : float, optional
        Pitch angle in degrees fixing the meridional travel direction: the
        arc-length-zero end is the one with the smallest projection onto
        ``d = (-sin Beta, cos Beta)``. ``Beta = 0`` puts arc length 0 at minimum
        r; ``Beta = +/-90`` at maximum / minimum x. Default 0.
    periodic : bool, optional
        If True (default), wrap theta modulo the pitch so the uniform theta
        window is fully populated. If False, span the raw cloud theta range and
        rely on the nearest-neighbour fallback outside the data hull.
    rtol : float, optional
        Relative tolerance for the periodic coverage check. With
        ``periodic=True`` the cloud theta span must be at least
        ``(1 - rtol) * pitch`` or a ``ValueError`` is raised. Default 0.05.

    Returns
    -------
    cut : Block, shape (ni, nj)
        Structured block with data interpolated onto the line-conforming grid.
    """
    ni, nj = interp_shape

    # Flatten triangle vertices to a point cloud, shape (ntri*3, nvar).
    triangle_data = unstructured_block._data
    _, _, nvar = triangle_data.shape
    points = triangle_data.reshape(-1, nvar)
    xr = points[:, :2].astype(np.float64)
    t_src = points[:, 2].astype(np.float64)
    variables = points[:, 3:]

    # Project the whole cloud onto the meridional line's principal direction
    # with a single SVD. The first right-singular vector is the line's dominant
    # direction; the projection s is effectively distance travelled along it and
    # is reused both to parametrise the reference line and to place every source
    # point in arc length.
    span = max(np.ptp(xr[:, 0]), np.ptp(xr[:, 1]))
    if span <= 0:
        raise ValueError("cut has no meridional extent")
    mean = xr.mean(0)
    _, _, Vt = np.linalg.svd(xr - mean, full_matrices=False)
    s = (xr - mean) @ Vt[0]

    # Orient so arc length 0 lies at the end with the smallest projection onto
    # the Beta reference direction d. The SVD sign is arbitrary, so flip s if the
    # minimum-s end disagrees with d.
    beta_rad = np.radians(Beta)
    d = np.array([-np.sin(beta_rad), np.cos(beta_rad)])
    if xr[s.argmin()] @ d > xr[s.argmax()] @ d:
        s = -s

    # Reference line: de-duplicate by meridional footprint (snap (x, r) to a grid
    # of size atol), order the nodes by the oriented projection, and accumulate
    # chord length into a normalised arc length zeta_ref.
    atol = 1e-6 * span
    key = np.round(xr / atol)
    _, idx = np.unique(key, axis=0, return_index=True)
    order = idx[np.argsort(s[idx])]
    xr_ref = xr[order]
    s_ref = s[order]
    if xr_ref.shape[0] < 2:
        raise ValueError("cut has fewer than 2 distinct meridional stations")
    seg = np.sqrt((np.diff(xr_ref, axis=0) ** 2).sum(1))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    zeta_ref = arc / arc[-1]

    # Per-point arc-length coordinate from the same projection.
    zeta_src = np.interp(s, s_ref, zeta_ref)

    # Target line nodes from cosine clustering along zeta (constant across j).
    zeta_t = util.cosine_cluster(ni).astype(np.float64)
    x_t = np.interp(zeta_t, zeta_ref, xr_ref[:, 0])
    r_t = np.interp(zeta_t, zeta_ref, xr_ref[:, 1])

    # Unfold theta to a normalised [0, 1] coordinate for well-conditioned
    # interpolation, and lay out the uniform target theta window.
    if periodic:
        pitch = float(unstructured_block.pitch)
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("periodic=True requires a finite positive pitch (set Nb)")
        span = float(np.ptp(t_src))
        if span < (1.0 - rtol) * pitch:
            raise ValueError(
                f"cut spans only {span:.4g} rad in theta, well under one pitch "
                f"{pitch:.4g}; periodic interpolation would fabricate data"
            )
        # Normalise theta by pitch WITHOUT wrapping: the source triangles were
        # connected in unwrapped theta, so a modulo here would fold the
        # theta=pitch boundary back onto 0 and turn last-cell triangles into
        # huge seam-straddling ones. The +/- one-period tiling below supplies
        # seam continuity instead.
        t0 = float(t_src.min())
        tn_src = (t_src - t0) / pitch
        theta_t = np.linspace(t0, t0 + pitch, nj)
        tn_t = (theta_t - t0) / pitch
        # Tile +/- one period so linear interpolation is continuous at the seam.
        src_pts = np.concatenate(
            [np.column_stack([zeta_src, tn_src + d]) for d in (-1.0, 0.0, 1.0)]
        )
        src_vars = np.concatenate([variables, variables, variables], axis=0)
    else:
        lo, hi = float(t_src.min()), float(t_src.max())
        width = hi - lo if hi > lo else 1.0
        tn_src = (t_src - lo) / width
        theta_t = np.linspace(lo, hi, nj)
        tn_t = (theta_t - lo) / width
        src_pts = np.column_stack([zeta_src, tn_src])
        src_vars = variables

    # Target points in the unfolded (zeta, theta_norm) unit space.
    ZT, TT = np.meshgrid(zeta_t, tn_t, indexing="ij")
    target_pts = np.column_stack([ZT.ravel(), TT.ravel()])

    # Interpolate on the block's own triangles (3 consecutive vertices each)
    # rather than re-triangulating the cloud. Reshape the per-vertex unfolded
    # coords and variables back into triangles, tiling +/- one period in the
    # periodic case so the seam stays continuous (mirrors the src_pts tiling).
    ntri_src = triangle_data.shape[0]
    nvar_var = variables.shape[1]
    tri_zeta = zeta_src.reshape(ntri_src, 3)
    tri_tn = tn_src.reshape(ntri_src, 3)
    tri_var = variables.reshape(ntri_src, 3, nvar_var)
    if periodic:
        tri_xy = np.concatenate(
            [np.stack([tri_zeta, tri_tn + d], axis=-1) for d in (-1.0, 0.0, 1.0)]
        )
        tri_var = np.concatenate([tri_var, tri_var, tri_var])
    else:
        tri_xy = np.stack([tri_zeta, tri_tn], axis=-1)

    # Linear interpolation in the Fortran kernel: rows returned NaN fell in no
    # triangle (outside the triangulated region, e.g. the zeta ends). Fill those
    # element-wise from the nearest source vertex via a KD-tree, matching the
    # previous scipy linear + nearest-fallback behaviour.
    fa = lambda x: np.asfortranarray(x, dtype=np.float32)  # noqa: E731
    interp = ember.fortran.tri_interp_linear(fa(tri_xy), fa(tri_var), fa(target_pts))
    tree = KDTree(np.ascontiguousarray(src_pts, dtype=np.float32))
    _, idx = tree.query(np.ascontiguousarray(target_pts, dtype=np.float32))
    near = src_vars[idx.ravel()]
    interp = np.where(np.isnan(interp), near, interp)

    # Assemble the structured output block.
    output_data = np.full((ni, nj, nvar), np.nan)
    output_data[..., 0] = x_t[:, None]
    output_data[..., 1] = r_t[:, None]
    output_data[..., 2] = theta_t[None, :]
    output_data[..., 3:] = interp.reshape(ni, nj, -1)

    result_block = unstructured_block.empty(shape=(ni, nj))
    result_block._data = output_data
    return result_block


def triangulate_to_unstructured(block):
    """Convert a structured 2D cut into a triangulated unstructured cut.

    Splits every structured quad face into two triangles, producing an
    unstructured block with ``ntri = 2 (ni - 1) (nj - 1)`` triangles.

    Parameters
    ----------
    block : Block, shape (ni, nj)
        2D structured block to triangulate.

    Returns
    -------
    cut : Block, shape (ntri, 3)
        Unstructured block with triangle-vertex data.
    """
    # Only work on 2D cuts
    assert block.ndim == 2

    # Every structured quad becomes two triangles:
    #
    # i,j+1 +----+ i+1, j+1
    #       |A / |
    #       | / B|
    #   i,j +----+ i+1, j
    #

    ni, nj = block.shape
    ntri = (ni - 1) * (nj - 1) * 2

    # Create index arrays for vectorized access
    i_indices, j_indices = np.meshgrid(range(ni - 1), range(nj - 1), indexing="ij")
    i_flat = i_indices.ravel()
    j_flat = j_indices.ravel()

    # Preallocate output data
    out = block.empty(shape=(ntri, 3))

    # Extract vertex data for all quads at once
    # Triangle A vertices: (i,j), (i,j+1), (i+1,j+1)
    v1_A = block._data[i_flat, j_flat, :]  # Shape: (nquads, nvar)
    v2_A = block._data[i_flat, j_flat + 1, :]
    v3_A = block._data[i_flat + 1, j_flat + 1, :]

    # Triangle B vertices: (i,j), (i+1,j+1), (i+1,j)
    v1_B = block._data[i_flat, j_flat, :]
    v2_B = block._data[i_flat + 1, j_flat + 1, :]
    v3_B = block._data[i_flat + 1, j_flat, :]

    # Assign triangle data - interleave A and B triangles

    # Triangle A data
    out._data[0::2, 0, :] = v1_A  # First vertex of triangle A
    out._data[0::2, 1, :] = v2_A  # Second vertex of triangle A
    out._data[0::2, 2, :] = v3_A  # Third vertex of triangle A

    # Triangle B data
    out._data[1::2, 0, :] = v1_B  # First vertex of triangle B
    out._data[1::2, 1, :] = v2_B  # Second vertex of triangle B
    out._data[1::2, 2, :] = v3_B  # Third vertex of triangle B

    out.set_triangulated(True)
    return out
