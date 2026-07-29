# coding: utf-8

"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from shapely import plotting
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from lib.IPPerfMonitor import IPPerfMonitor


class CollisionChecker(object):

    def __init__(
        self,
        scene: Dict[str, BaseGeometry],
        limits: List[List[float]] | None = None,
        statistic: Any = None,
    ):
        self.scene: Dict[str, BaseGeometry] = scene
        self.limits: List[List[float]] = (
            [[0.0, 22.0], [0.0, 22.0]] if limits is None else limits
        )
        self.statistic = statistic

    def getDim(self) -> int:
        """Return dimension of Environment (Shapely should currently always be 2)."""
        return 2

    def getEnvironmentLimits(self):
        """Return limits of Environment."""
        return list(self.limits)

    def isInLimits(self, pos: np.ndarray) -> bool:
        """Return whether a configuration lies inside the configured limits."""
        assert len(pos) == self.getDim()
        return (
            self.limits[0][0] <= pos[0] <= self.limits[0][1]
            and self.limits[1][0] <= pos[1] <= self.limits[1][1]
        )

    def pointInCollision(self, pos: np.ndarray) -> bool:
        """Return whether a configuration is invalid.
        Collision or outside limits -> True
        Free and inside limits -> False
        """
        assert len(pos) == self.getDim()
        if not self.isInLimits(pos):
            return True
        for _, value in self.scene.items():
            if value.intersects(Point(pos[0], pos[1])):
                return True
        return False

    @IPPerfMonitor
    def lineInCollision(
        self,
        startPos: np.ndarray,
        endPos: np.ndarray,
        steps: int | None = 40,
    ) -> bool:
        """Check a line by sampling intermediate points.
        This illustrates local planner discretization, but it may miss thin obstacles.
        """
        assert len(startPos) == self.getDim()
        assert len(endPos) == self.getDim()
        assert steps is not None and steps > 0
        dim = self.getDim()
        for i in range(steps + 1):
            t = i / steps
            testPoint = np.asarray(
                [
                    startPos[d] + t * (endPos[d] - startPos[d])
                    for d in range(dim)
                ]
            )
            if self.pointInCollision(testPoint):
                return True
        return False

    def lineInCollisionExact(
        self,
        startPos: np.ndarray,
        endPos: np.ndarray,
    ) -> bool:
        """Check whether the exact line segment from startPos to endPos collides."""
        assert len(startPos) == self.getDim()
        assert len(endPos) == self.getDim()
        if self.pointInCollision(startPos) or self.pointInCollision(endPos):
            return True
        line = LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])
        for _, value in self.scene.items():
            if value.intersects(line):
                return True
        return False

    def drawObstacles(self, ax: plt.Axes | None):
        for _, value in self.scene.items():
            plotting.plot_polygon(value, add_points=False, ax=ax, color="red")  # type: ignore
