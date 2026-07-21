# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import random
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from lib.IPEnvironment import CollisionChecker
from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPlanerBase import PlanerBase
from PointProjection import Point


class PRMBase(PlanerBase):

    def __init__(self, collChecker: CollisionChecker):
        super(PRMBase, self).__init__(collChecker)
        self.graph: nx.Graph[Any] = nx.Graph()
        self.lastGeneratedNodeNumber: int = 0

    def _getRandomPosition(self) -> np.ndarray:
        limits = self.collisionChecker.getEnvironmentLimits()
        pos = [random.uniform(limit[0], limit[1]) for limit in limits]
        return np.array(pos)

    def _getRandomFreePosition(self) -> np.ndarray:
        pos = self._getRandomPosition()
        while self.collisionChecker.pointInCollision(pos):
            pos = self._getRandomPosition()
        return pos

    def createNode(self, pos: Point, **kwargs: Any) -> int:
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=pos, **kwargs)
        self.lastGeneratedNodeNumber += 1
        return self.lastGeneratedNodeNumber - 1

    @IPPerfMonitor
    def planPath(
        self,
        startList: List[np.ndarray],
        goalList: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        raise NotImplementedError("planPath is not implemented in PRMBase")

    def getNodePositions(self) -> List[np.ndarray]:
        return list(nx.get_node_attributes(self.graph, "pos").values())  # type: ignore

    def getShortestPathFromStartToGoal(self) -> List[np.ndarray]:
        try:
            return nx.shortest_path(self.graph, "start", "goal")  # type: ignore
        except nx.NetworkXNoPath:
            print("No path found from start to goal")
            return []
