from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPRMBase import PRMBase
from PointProjection import Point


class BiRRTConfig(TypedDict):
    numberOfGeneratedNodes: int
    # testGoalAfterNumberOfNodes: int
    stepSize: float
    balanceTrees: bool
    sampleGoalProbability: float
    collisionDetectionSteps: int
    maxIterations: int
    # orthogonalityMargin: float


class BiRRT(PRMBase):

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

    def getNextMode(
        self, balanceTrees: bool
    ) -> Union[Literal["forward"], Literal["backward"]]:
        if balanceTrees:
            if (
                self.getForwardSubtree().number_of_nodes()
                <= self.getBackwardSubtree().number_of_nodes()
            ):
                return "forward"
            return "backward"
        else:
            if not hasattr(self, "_mode"):
                self._mode = "forward"
                return self._mode
            else:
                self._mode = (
                    "backward" if self._mode == "forward" else "forward"
                )
                return self._mode

    def sampleNextPosition(
        self,
        mode: Union[Literal["forward"], Literal["backward"]],
        sampleGoalProbability: float,
    ) -> np.ndarray:
        if sampleGoalProbability > np.random.rand():
            goalNode = (
                self.graph.nodes[1]
                if mode == "forward"
                else self.graph.nodes[0]
            )
            return np.asarray(goalNode["pos"])  # type: ignore
        return self._getRandomFreePosition()

    def stepTowardCandidate(
        self,
        candidatePos: np.ndarray,
        nearestNeighborPos: np.ndarray,
        stepSize: float,
    ) -> np.ndarray:
        diff = candidatePos - nearestNeighborPos
        dist = np.linalg.norm(diff)
        thisStep = min(dist, stepSize)
        return nearestNeighborPos + (diff / dist) * thisStep

    def setup(self, startList: List[Point], goalList: List[Point]):
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0

        checkedStartList, checkedGoalList = self._checkStartGoal(
            startList, goalList
        )
        self.start_pos = checkedStartList[0]
        self.goal_pos = checkedGoalList[0]

        self.start_node_id = 0
        self.goal_node_id = 1

        self.createNode(self.start_pos, mode="forward")
        self.createNode(self.goal_pos, mode="backward")

    @IPPerfMonitor
    def planPath(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        startList: List[np.ndarray],
        goalList: List[np.ndarray],
        config: BiRRTConfig,
    ) -> Tuple[List[np.ndarray], Optional[str]]:
        """
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
        """
        self.setup(startList, goalList)

        numIterations: int = 0

        stepSize = config.get("stepSize", np.inf)
        maxIterations: int = config.get("maxIterations", 10000)
        sampleGoalProbability: float = config.get("sampleGoalProbability", 0.0)
        collisionDetectionSteps = config.get("collisionDetectionSteps", 40)
        numberOfGeneratedNodes = config.get("numberOfGeneratedNodes", 200)
        balanceTrees: bool = config.get("balanceTrees", True)

        while self.lastGeneratedNodeNumber < numberOfGeneratedNodes:
            if numIterations >= maxIterations:
                return [], "max_iterations_reached"
            numIterations += 1

            mode = self.getNextMode(balanceTrees)

            random_free_pos = self.sampleNextPosition(
                mode, sampleGoalProbability
            )

            nearest_neighbor_idx, nearest_neighbor = (
                self.getNearestNeighborInSubtree(random_free_pos, mode)
            )

            candidate_step = self.stepTowardCandidate(
                random_free_pos, nearest_neighbor["pos"], stepSize
            )

            if not self.collisionChecker.lineInCollision(
                nearest_neighbor["pos"],
                candidate_step,
                steps=collisionDetectionSteps,
            ):
                new_node_id = self.createNode(candidate_step, mode=mode)

                self.graph.add_edge(nearest_neighbor_idx, new_node_id)

                # test new_node connection to the nearest neighbor in the other subtree
                (
                    nearest_neighbor_from_opposite_tree_id,
                    nearestOtherNeighbor,
                ) = self.getNearestNeighborInSubtree(
                    candidate_step,
                    "backward" if mode == "forward" else "forward",
                )
                nearest_neighbor_from_opposite_tree_pos = nearestOtherNeighbor[
                    "pos"
                ]
                conn_dist: float = np.linalg.norm(  # type: ignore
                    candidate_step - nearest_neighbor_from_opposite_tree_pos
                )
                if (
                    not self.collisionChecker.lineInCollision(
                        nearestOtherNeighbor["pos"],
                        candidate_step,
                        steps=collisionDetectionSteps,
                    )
                    and conn_dist <= stepSize
                ):
                    self.graph.add_edge(
                        new_node_id,
                        nearest_neighbor_from_opposite_tree_id,
                    )
                    mapping = {
                        self.start_node_id: "start",
                        self.goal_node_id: "goal",
                    }
                    self.graph = nx.relabel_nodes(self.graph, mapping)
                    return self.getShortestPathFromStartToGoal(), None

        return [], "max_nodes_reached"
