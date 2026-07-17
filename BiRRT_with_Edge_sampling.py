from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from lib.IPEnvironment import CollisionChecker
from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPRMBase import PRMBase
from PointProjection import projectPointOnEdge
from RRT_with_Edge_sampling import Point, ProjectedPoint


class BiRRTEdgeConfig(TypedDict):
    numberOfGeneratedNodes: int
    # testGoalAfterNumberOfNodes: int
    stepSize: float
    balanceTrees: bool
    sampleGoalProbability: float
    collisionDetectionSteps: int
    maxIterations: int
    orthogonalityMargin: float


class BiRRTEdge(PRMBase):

    def __init__(self, collChecker: CollisionChecker):
        """
        _collChecker: the collision checker interface
        """
        super(BiRRTEdge, self).__init__(collChecker)
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

    def divide_edge(self, edgePoint: Point, start_Id: int, end_Id: int):
        assert (
            self.graph.nodes[start_Id]["mode"]
            == self.graph.nodes[end_Id]["mode"]
        ), "Both nodes must have the same mode"
        mode = self.graph.nodes[start_Id]["mode"]
        # Remove old edge
        self.graph.remove_edge(start_Id, end_Id)

        # Add new edges and point
        self.graph.add_node(
            self.lastGeneratedNodeNumber, pos=edgePoint, mode=mode
        )

        self.graph.add_edge(start_Id, self.lastGeneratedNodeNumber)
        self.graph.add_edge(self.lastGeneratedNodeNumber, end_Id)

        self.lastGeneratedNodeNumber += 1

        return self.lastGeneratedNodeNumber - 1

    def getEdgesOfSubtree(
        self, mode: Union[Literal["forward"], Literal["backward"]]
    ) -> List[Tuple[int, int]]:
        match mode:
            case "forward":
                subtree = self.getForwardSubtree()
            case "backward":
                subtree = self.getBackwardSubtree()
        return list(subtree.edges())

    def sampleNextPosition(
        self,
        mode: Union[Literal["forward"], Literal["backward"]],
        sampleGoalProbability: float,
    ) -> np.ndarray:
        if sampleGoalProbability > np.random.rand():
            goalNode = (
                self.graph.nodes[0]
                if mode == "forward"
                else self.graph.nodes[1]
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

    @IPPerfMonitor
    def planPath(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        startList: List[np.ndarray],
        goalList: List[np.ndarray],
        config: BiRRTEdgeConfig,
    ) -> Tuple[List[np.ndarray], Optional[str]]:
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
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
            pos=np.asarray(checkedStartList[0]),
            mode="forward",
        )
        self.lastGeneratedNodeNumber += 1

        self.graph.add_node(
            1,
            pos=np.asarray(checkedGoalList[0]),
            mode="backward",
        )
        self.lastGeneratedNodeNumber += 1

        stepSize = config.get("stepSize", np.inf)

        numIterations: int = 0
        maxIterations: int = config.get("maxIterations", 10000)
        sampleGoalProbability: float = config.get("sampleGoalProbability", 0.0)
        orthogonalityMargin: float = config.get("orthogonalityMargin", 0.0)

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

            edges = self.getEdgesOfSubtree(mode)

            # handle case when there are no edges yet (only start node / goal node in backward mode)
            if not edges:
                begin_pos = (
                    self.graph.nodes[0]["pos"]
                    if mode == "forward"
                    else self.graph.nodes[1]["pos"]
                )
                candidate_step = self.stepTowardCandidate(
                    random_free_pos, begin_pos, stepSize
                )
                if not self.collisionChecker.lineInCollision(
                    begin_pos, candidate_step, steps=collisionDetectionSteps
                ):
                    self.graph.add_node(
                        self.lastGeneratedNodeNumber,
                        pos=candidate_step,
                        mode=mode,
                    )
                    self.graph.add_edge(
                        0 if mode == "forward" else 1,
                        self.lastGeneratedNodeNumber,
                    )
                    self.lastGeneratedNodeNumber += 1
                continue

            proj_points: List[ProjectedPoint] = []
            for u, v in edges:
                u_pos = self.graph.nodes[u]["pos"]
                v_pos = self.graph.nodes[v]["pos"]
                proj = projectPointOnEdge(
                    random_free_pos, u_pos, v_pos, orthogonalityMargin
                )
                p = ProjectedPoint(
                    point=proj[0], t=proj[1], edge_start_id=u, edge_end_id=v
                )
                proj_points.append(p)
            pos_list = [p.point for p in proj_points]
            kd_tree = KDTree(pos_list)
            _, nearest_index_in_projection_points = kd_tree.query(
                random_free_pos, k=1
            )
            nearest_proj_point = proj_points[
                nearest_index_in_projection_points
            ]

            candidate_step = self.stepTowardCandidate(
                random_free_pos, nearest_proj_point.point, stepSize
            )

            if not self.collisionChecker.lineInCollision(
                nearest_proj_point.point,
                candidate_step,
                steps=collisionDetectionSteps,
            ):
                self.graph.add_node(
                    self.lastGeneratedNodeNumber,
                    pos=candidate_step,
                    mode=mode,
                )
                candidate_node_id = self.lastGeneratedNodeNumber
                self.lastGeneratedNodeNumber += 1

                if (
                    nearest_proj_point.t > orthogonalityMargin
                    and nearest_proj_point.t < 1 - orthogonalityMargin
                ):
                    new_node_id = self.divide_edge(
                        nearest_proj_point.point,
                        nearest_proj_point.edge_start_id,
                        nearest_proj_point.edge_end_id,
                    )
                    self.graph.add_edge(new_node_id, candidate_node_id)
                    # Check if graph is still acyclic and connected
                    if not nx.is_tree(
                        self.getForwardSubtree()
                        if mode == "forward"
                        else self.getBackwardSubtree()
                    ):
                        raise Exception(
                            f"The subgraph {mode} is either not connected or not acyclic",
                            (
                                self.getForwardSubtree()
                                if mode == "forward"
                                else self.getBackwardSubtree()
                            ),
                        )
                elif nearest_proj_point.t <= orthogonalityMargin:
                    self.graph.add_edge(
                        nearest_proj_point.edge_start_id,
                        candidate_node_id,
                    )
                else:
                    self.graph.add_edge(
                        nearest_proj_point.edge_end_id,
                        candidate_node_id,
                    )

            nearest_neighbor_idx, nearest_neighbor = (
                self.getNearestNeighborInSubtree(random_free_pos, mode)
            )

            candidate_step = self.stepTowardCandidate(
                random_free_pos, nearest_neighbor["pos"], stepSize
            )

            # test new_node connection to the nearest neighbor in the other subtree
            if not self.collisionChecker.lineInCollision(
                nearest_neighbor["pos"],
                candidate_step,
                steps=collisionDetectionSteps,
            ):
                self.graph.add_node(
                    self.lastGeneratedNodeNumber,
                    pos=candidate_step,
                    mode=mode,
                )

                self.graph.add_edge(
                    nearest_neighbor_idx, self.lastGeneratedNodeNumber
                )
                nearestOtherNeighborIdx, nearestOtherNeighbor = (
                    self.getNearestNeighborInSubtree(
                        candidate_step,
                        "backward" if mode == "forward" else "forward",
                    )
                )
                conn_dist = np.linalg.norm(  # type: ignore
                    candidate_step - nearestOtherNeighbor["pos"]
                )
                if (
                    not self.collisionChecker.lineInCollision(
                        nearestOtherNeighbor["pos"],
                        candidate_step,
                    )
                    and conn_dist <= stepSize
                ):
                    self.graph.add_edge(
                        self.lastGeneratedNodeNumber,
                        nearestOtherNeighborIdx,
                    )
                    mapping = {0: "start", 1: "goal"}
                    self.graph = nx.relabel_nodes(self.graph, mapping)
                    return self.getShortestPathFromStartToGoal(), None
                self.lastGeneratedNodeNumber += 1

        return [], "max_nodes_reached"
