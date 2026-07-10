from typing import Any, Dict, List

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from lib.IPEnvironment import CollisionChecker
from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPRMBase import PRMBase


class RRTEdgeSampling(PRMBase):

    def __init__(self, _collChecker: CollisionChecker):
        """
        _collChecker: the collision checker interface
        """
        super(RRTEdgeSampling, self).__init__(_collChecker)
        self.lastGeneratedNodeNumber: int = 0

    @IPPerfMonitor
    def planPath(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        startList: List[np.ndarray],
        goalList: List[np.ndarray],
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:
            config["numberOfGeneratedNodes"] = 500
            config["testGoalAfterNumberOfNodes"]  = 10
        """
        # 0. reset
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0

        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(
            startList, goalList
        )

        # 2. add start and goal to graph
        self.graph.add_node(
            self.lastGeneratedNodeNumber, pos=checkedStartList[0]
        )
        self.lastGeneratedNodeNumber += 1

        while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:

            posList = self.getNodePositions()
            kdTree = KDTree(posList)

            if (
                self.lastGeneratedNodeNumber
                % config["testGoalAfterNumberOfNodes"]
            ) == 0:

                result = kdTree.query(checkedGoalList[0], k=1)

                if not self.collisionChecker.lineInCollision(
                    self.graph.nodes[result[1]]["pos"], checkedGoalList[0]
                ):
                    self.graph.add_node("goal", pos=checkedGoalList[0])
                    self.graph.add_edge(result[1], "goal")
                    mapping = {0: "start"}
                    self.graph = nx.relabel_nodes(self.graph, mapping)

                    return self.getShortestPathFromStartToGoal()

            pos = self._getRandomFreePosition()

            # for every node in graph find nearest neigbhours

            result = kdTree.query(pos, k=1)
            if None:
                raise Exception(
                    "Something went wrong regarding nearest neighbours"
                )
            # print result
            if not self.collisionChecker.lineInCollision(
                self.graph.nodes[result[1]]["pos"], pos
            ):
                self.graph.add_node(self.lastGeneratedNodeNumber, pos=pos)
                self.graph.add_edge(result[1], self.lastGeneratedNodeNumber)
                self.lastGeneratedNodeNumber += 1

        return []
