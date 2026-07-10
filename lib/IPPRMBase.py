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


class PRMBase(PlanerBase):

    def __init__(self, collChecker: CollisionChecker):
        super(PRMBase, self).__init__(collChecker)
        self.graph: nx.Graph[Any] = nx.Graph()

    def _getRandomPosition(self):
        limits = self.collisionChecker.getEnvironmentLimits()
        pos = [random.uniform(limit[0], limit[1]) for limit in limits]
        return pos

    @IPPerfMonitor
    def _getRandomFreePosition(self):
        pos = self._getRandomPosition()
        while self.collisionChecker.pointInCollision(pos):
            pos = self._getRandomPosition()
        return pos

    @IPPerfMonitor
    def planPath(
        self,
        startList: List[np.ndarray],
        goalList: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        raise NotImplementedError("planPath is not implemented in PRMBase")

    def getGraph(self) -> nx.Graph[Any]:
        return self.graph

    def getNodePositions(self) -> List[np.ndarray]:
        return list(nx.get_node_attributes(self.graph, "pos").values())  # type: ignore

    def getShortestPathFromStartToGoal(self) -> List[np.ndarray]:
        return nx.shortest_path(self.graph, "start", "goal")  # type: ignore
