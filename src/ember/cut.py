"""Marching cubes algorithm and 3D structured grid cutting utilities.

This module implements the marching cubes algorithm for extracting isosurfaces from 3D
structured grids, along with utilities for creating both structured meridional cuts and
unstructured triangulated cuts. The marching cubes implementation uses precomputed lookup
tables (EDGETABLE and TRITABLE) to efficiently determine which cell edges intersect an
isosurface and how to triangulate those intersections. Key functionality includes extracting
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

# Which edges are cut by the isosurface? 256 possible cases, each corresponding to
# a 12-bit number, each bit corresponds to an edge
_EDGETABLE = np.array(
    [
        0x0,
        0x109,
        0x203,
        0x30A,
        0x406,
        0x50F,
        0x605,
        0x70C,
        0x80C,
        0x905,
        0xA0F,
        0xB06,
        0xC0A,
        0xD03,
        0xE09,
        0xF00,
        0x190,
        0x99,
        0x393,
        0x29A,
        0x596,
        0x49F,
        0x795,
        0x69C,
        0x99C,
        0x895,
        0xB9F,
        0xA96,
        0xD9A,
        0xC93,
        0xF99,
        0xE90,
        0x230,
        0x339,
        0x33,
        0x13A,
        0x636,
        0x73F,
        0x435,
        0x53C,
        0xA3C,
        0xB35,
        0x83F,
        0x936,
        0xE3A,
        0xF33,
        0xC39,
        0xD30,
        0x3A0,
        0x2A9,
        0x1A3,
        0xAA,
        0x7A6,
        0x6AF,
        0x5A5,
        0x4AC,
        0xBAC,
        0xAA5,
        0x9AF,
        0x8A6,
        0xFAA,
        0xEA3,
        0xDA9,
        0xCA0,
        0x460,
        0x569,
        0x663,
        0x76A,
        0x66,
        0x16F,
        0x265,
        0x36C,
        0xC6C,
        0xD65,
        0xE6F,
        0xF66,
        0x86A,
        0x963,
        0xA69,
        0xB60,
        0x5F0,
        0x4F9,
        0x7F3,
        0x6FA,
        0x1F6,
        0xFF,
        0x3F5,
        0x2FC,
        0xDFC,
        0xCF5,
        0xFFF,
        0xEF6,
        0x9FA,
        0x8F3,
        0xBF9,
        0xAF0,
        0x650,
        0x759,
        0x453,
        0x55A,
        0x256,
        0x35F,
        0x55,
        0x15C,
        0xE5C,
        0xF55,
        0xC5F,
        0xD56,
        0xA5A,
        0xB53,
        0x859,
        0x950,
        0x7C0,
        0x6C9,
        0x5C3,
        0x4CA,
        0x3C6,
        0x2CF,
        0x1C5,
        0xCC,
        0xFCC,
        0xEC5,
        0xDCF,
        0xCC6,
        0xBCA,
        0xAC3,
        0x9C9,
        0x8C0,
        0x8C0,
        0x9C9,
        0xAC3,
        0xBCA,
        0xCC6,
        0xDCF,
        0xEC5,
        0xFCC,
        0xCC,
        0x1C5,
        0x2CF,
        0x3C6,
        0x4CA,
        0x5C3,
        0x6C9,
        0x7C0,
        0x950,
        0x859,
        0xB53,
        0xA5A,
        0xD56,
        0xC5F,
        0xF55,
        0xE5C,
        0x15C,
        0x55,
        0x35F,
        0x256,
        0x55A,
        0x453,
        0x759,
        0x650,
        0xAF0,
        0xBF9,
        0x8F3,
        0x9FA,
        0xEF6,
        0xFFF,
        0xCF5,
        0xDFC,
        0x2FC,
        0x3F5,
        0xFF,
        0x1F6,
        0x6FA,
        0x7F3,
        0x4F9,
        0x5F0,
        0xB60,
        0xA69,
        0x963,
        0x86A,
        0xF66,
        0xE6F,
        0xD65,
        0xC6C,
        0x36C,
        0x265,
        0x16F,
        0x66,
        0x76A,
        0x663,
        0x569,
        0x460,
        0xCA0,
        0xDA9,
        0xEA3,
        0xFAA,
        0x8A6,
        0x9AF,
        0xAA5,
        0xBAC,
        0x4AC,
        0x5A5,
        0x6AF,
        0x7A6,
        0xAA,
        0x1A3,
        0x2A9,
        0x3A0,
        0xD30,
        0xC39,
        0xF33,
        0xE3A,
        0x936,
        0x83F,
        0xB35,
        0xA3C,
        0x53C,
        0x435,
        0x73F,
        0x636,
        0x13A,
        0x33,
        0x339,
        0x230,
        0xE90,
        0xF99,
        0xC93,
        0xD9A,
        0xA96,
        0xB9F,
        0x895,
        0x99C,
        0x69C,
        0x795,
        0x49F,
        0x596,
        0x29A,
        0x393,
        0x99,
        0x190,
        0xF00,
        0xE09,
        0xD03,
        0xC0A,
        0xB06,
        0xA0F,
        0x905,
        0x80C,
        0x70C,
        0x605,
        0x50F,
        0x406,
        0x30A,
        0x203,
        0x109,
        0x0,
    ],
    dtype=int,
)

# How to form triangles from the cut edges
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


def _cube_index(d):
    """For a 3D array of signed distances, get cube indices."""
    ni, nj, nk = d.shape
    ind = np.zeros((ni - 1, nj - 1, nk - 1), dtype=int)
    for v in range(8):
        ind[d[_vijk(d.shape, v)] < 0.0] |= 2**v
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

    ni, nj, nk, nvar = data.shape

    # Find an index into edge table to see which edges are cut
    icube = _cube_index(dist)
    edge_index = _EDGETABLE[icube]

    # Now treat each cell individually
    # Most cells are not cut so for loop is not too bad
    triangles = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            for k in range(nk - 1):
                # Skip uncut cells
                if not edge_index[i, j, k]:
                    continue

                # Preallocate for vertices that could be on any of 12 edges
                cut_edges = np.full((12, nvar), np.nan)

                # Loop over the vertices
                for e in range(12):
                    # If the edge index contains the bit
                    if edge_index[i, j, k] & 2**e:
                        # Get start and end indices for the edge
                        ijk_st, ijk_en = _eijk(i, j, k, e)

                        # Slice spatial dimensions, variables are in last axis
                        data_st = data[ijk_st]  # Shape (nvar,)
                        data_en = data[ijk_en]  # Shape (nvar,)

                        # Perform linear interpolation
                        frac = -dist[ijk_st] / (dist[ijk_en] - dist[ijk_st])
                        assert (frac >= 0.0) and (frac <= 1.0)
                        cut_edges[e] = data_st + (data_en - data_st) * frac

                # We have found all the vertices we will need
                # Now use cube_index to look up in TRITABLE how to
                # assemble into triangles
                triangle_index = _TRITABLE[icube[i, j, k]]

                # Loop over the triangle indices in threes
                for itri in range(0, len(triangle_index), 3):
                    # Sentinel value indices no more triangles
                    if triangle_index[itri] == -1:
                        break

                    # Pull out the cut edges for this triangle
                    triangles.append(cut_edges[triangle_index[itri : itri + 3]])

    if triangles:
        return np.stack(triangles)
    else:
        return None


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

    # Import at function level to avoid circular imports
    from ember.block import Block

    # Convert single Block to list for consistent handling
    if isinstance(grid, Block):
        grid = [grid]

    # Loop over blocks
    triangles = []
    last_block = None
    for block in grid:
        # Check if cut line intersects this block's domain
        xr_coords = block.xrt[..., :2]

        # Evaluate signed distance for all points in the block
        # Note: need to import at function level to avoid circular imports
        from ember.util import signed_distance

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

    # Import at function level to avoid circular imports
    from ember.util import signed_distance
    from ember.block import Block
    import ember.grid

    # Convert single Block to list for consistent handling
    if isinstance(grid, Block):
        grid = [grid]

    cut_blocks = []

    for block in grid:
        # Get meridional coordinates
        xr_coords = block.xrt[..., :2]  # Extract (x, r) coordinates

        # Get signed distance
        dist = signed_distance(xr_cut, xr_coords)

        # Check for intersection
        if np.all(dist >= 0) or np.all(dist <= 0):
            continue

        ni, nj, nk = block.shape

        # Check if cut intersects this block
        if np.all(dist >= 0) or np.all(dist <= 0):
            continue

        # Preallocate cut data for this block
        cut_data = np.full((ni, nk, block._data.shape[-1]), np.nan)

        # Find intersections along j-direction for each (i, k) line
        for i in range(ni):
            for k in range(nk):
                # Get distance along j-direction for this (i, k) line
                dist_line = dist[i, :, k]

                # Find where distance changes sign
                sign_changes = np.where(np.diff(np.sign(dist_line)) != 0)[0]

                if len(sign_changes) > 0:
                    # Take the first sign change
                    j_cut = sign_changes[0]

                    # Avoid division by zero
                    d1, d2 = dist_line[j_cut], dist_line[j_cut + 1]
                    if abs(d2 - d1) < 1e-12:
                        continue

                    # Linear interpolation
                    frac = -d1 / (d2 - d1)

                    # Clamp fraction to [0, 1]
                    frac = max(0.0, min(1.0, frac))

                    # Interpolate all data variables
                    data1 = block._data[i, j_cut, k, :]
                    data2 = block._data[i, j_cut + 1, k, :]
                    cut_data[i, k, :] = data1 + (data2 - data1) * frac

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
    from ember import util
    from pykdtree.kdtree import KDTree
    import ember.fortran

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
