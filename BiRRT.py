from typing import Any, Dict, List, Literal, Tuple, Union

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from lib.IPEnvironment import CollisionChecker
from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPRMBase import PRMBase


class BiRRT(PRMBase):

    def __init__(self, collChecker: CollisionChecker):
        """
        _collChecker: the collision checker interface
        """
        super(BiRRT, self).__init__(collChecker)
        self.lastGeneratedNodeNumber: int = 0

    def getForwardSubtree(self) -> nx.Graph[Any]:
        forwardGraphNodes = [
            n
            for n, v in self.graph.nodes(data=True)
            if v.get("mode") == "forward"
        ]
        return self.graph.subgraph(forwardGraphNodes)  # type: ignore

    def getBackwardSubtree(self) -> nx.Graph[Any]:
        backwardGraphNodes = [
            n
            for n, v in self.graph.nodes(data=True)
            if v.get("mode") == "backward"
        ]
        return self.graph.subgraph(backwardGraphNodes)  # type: ignore

    def getNearestNeighborInSubtree(
        self,
        pos: np.ndarray,
        mode: Union[Literal["forward"], Literal["backward"]],
    ) -> Tuple[int, Dict[str, Any]]:
        match mode:
            case "forward":
                subtree = self.getForwardSubtree()
            case "backward":
                subtree = self.getBackwardSubtree()

        pos_list = list(nx.get_node_attributes(subtree, "pos").values())  # type: ignore
        kd_tree = KDTree(pos_list)
        _, nearest_index_in_subtree = kd_tree.query(pos, k=1)
        nearest_node = list(subtree.nodes(data=True))[nearest_index_in_subtree]
        return nearest_node

    def getNextMode(self) -> Union[Literal["forward"], Literal["backward"]]:
        if (
            self.getForwardSubtree().number_of_nodes()
            <= self.getBackwardSubtree().number_of_nodes()
        ):
            return "forward"
        return "backward"

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
            0,
            pos=checkedStartList[0],
            mode="forward",
        )
        self.lastGeneratedNodeNumber += 1

        self.graph.add_node(
            1,
            pos=checkedGoalList[0],
            mode="backward",
        )
        self.lastGeneratedNodeNumber += 1

        while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:

            mode = self.getNextMode()

            new_candidate_pos = self._getRandomFreePosition()

            # for every node in graph find nearest neigbhours with same mode
            nearest_neighbor_idx, nearest_neighbor = (
                self.getNearestNeighborInSubtree(new_candidate_pos, mode)
            )
            # result = fullKdTree.query(pos, k=1)
            # print result
            if not self.collisionChecker.lineInCollision(
                nearest_neighbor["pos"],
                new_candidate_pos,
            ):
                self.graph.add_node(
                    self.lastGeneratedNodeNumber,
                    pos=new_candidate_pos,
                    mode=mode,
                )

                self.graph.add_edge(
                    nearest_neighbor_idx, self.lastGeneratedNodeNumber
                )
                nearestOtherNeighborIdx, nearestOtherNeighbor = (
                    self.getNearestNeighborInSubtree(
                        new_candidate_pos,
                        "backward" if mode == "forward" else "forward",
                    )
                )
                if not self.collisionChecker.lineInCollision(
                    nearestOtherNeighbor["pos"],
                    new_candidate_pos,
                ):
                    self.graph.add_edge(
                        self.lastGeneratedNodeNumber,
                        nearestOtherNeighborIdx,
                    )
                    mapping = {0: "start", 1: "goal"}
                    self.graph = nx.relabel_nodes(self.graph, mapping)
                    return self.getShortestPathFromStartToGoal()
                self.lastGeneratedNodeNumber += 1

        return []
