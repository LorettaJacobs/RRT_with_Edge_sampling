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
    if len(edgeStart) != len(edgeEnd) and len(edgeStart) != len(p):
        raise Exception("No valid parameters for projection")

    lineVec: Vector = np.subtract(edgeEnd, edgeStart)
    pointVec: Vector = np.subtract(p, edgeStart)

    t: float = np.dot(pointVec, lineVec) / np.dot(lineVec, lineVec)

    if t < orthogonalityMargin:
        return (edgeStart, t)
    elif t > 1 - orthogonalityMargin:
        return (edgeEnd, t)
    else:
        p_proj = edgeStart + t * lineVec
        return (p_proj, t)
