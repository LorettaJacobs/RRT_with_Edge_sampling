"""
Autor: Alisa Hummel
"""

from typing import Tuple

import numpy as np

type Point = np.ndarray
type Vector = np.ndarray


def projectPointOnEdge(
    p: Point,
    edgeStart: Point,
    edgeEnd: Point,
    orthogonalityMargin: float = 0.0,
) -> Tuple[Point, float]:
    if len(edgeStart) != len(edgeEnd) or len(edgeStart) != len(p):
        raise Exception("No valid parameters for projection")

    # points should not be inf or nan
    if (
        not np.isfinite(edgeStart).all()
        or not np.isfinite(edgeEnd).all()
        or not np.isfinite(p).all()
    ):
        raise Exception(
            "No valid parameters for projection, points are not finite",
            edgeStart,
            edgeEnd,
            p,
        )

    lineVec: Vector = np.subtract(edgeEnd, edgeStart)
    pointVec: Vector = np.subtract(p, edgeStart)

    t: float = np.dot(pointVec, lineVec) / np.dot(lineVec, lineVec)

    if not np.isfinite(t):
        raise Exception(
            "Projection failed, t is not finite",
            lineVec,
            pointVec,
            edgeStart,
            edgeEnd,
            p,
            t,
        )

    if t < orthogonalityMargin:
        return (edgeStart, t)
    elif t > 1 - orthogonalityMargin:
        return (edgeEnd, t)
    else:
        p_proj = edgeStart + t * lineVec
        assert np.isfinite(p_proj).all()
        return (p_proj, t)
