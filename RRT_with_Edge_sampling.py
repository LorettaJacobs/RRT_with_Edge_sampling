"""
Autor: Ole Hocker
"""

from typing import List, NamedTuple, Optional, Tuple, TypedDict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPRMBase import PRMBase
from PointProjection import projectPointOnEdge


class RRTEdgeConfig(TypedDict):
    numberOfGeneratedNodes: int
    testGoalAfterNumberOfNodes: int
    stepSize: float
    # balanceTree: bool
    sampleGoalProbability: float
    collisionDetectionSteps: int
    maxIterations: int
    orthogonalityMargin: float


class ProjectedPoint(NamedTuple):
    point: np.ndarray
    t: float
    edge_start_id: int
    edge_end_id: int


type Point = np.ndarray


class RRTEdge(PRMBase):

    @IPPerfMonitor
    def divide_edge(self, edgePoint: Point, start_Id: int, end_Id: int):
        # Remove old edge
        self.graph.remove_edge(start_Id, end_Id)

        # Add new edges and point
        projected_node_id = self.createNode(edgePoint, type="projected")

        self.graph.add_edge(start_Id, projected_node_id)
        self.graph.add_edge(projected_node_id, end_Id)

        return projected_node_id

    def sampleNextPosition(
        self,
        sampleGoalProbability: float,
    ) -> np.ndarray:
        if sampleGoalProbability > np.random.rand():
            return self.goal_pos
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
        self.start_pos = np.asarray(checkedStartList[0])
        self.goal_pos = np.asarray(checkedGoalList[0])

        self.start_node_id = 0
        # self.goal_node_id = 1

        self.createNode(self.start_pos, type="start")
        # self.createNode(self.goal_pos, mode="backward")

    @IPPerfMonitor
    def planPath(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        startList: List[np.ndarray],
        goalList: List[np.ndarray],
        config: RRTEdgeConfig,
    ) -> Tuple[List[np.ndarray], Optional[str]]:
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
        """
        self.setup(startList, goalList)

        numIterations: int = 0
        numGoalTests: int = 0

        stepSize = config.get("stepSize", np.inf)
        maxIterations: int = config.get("maxIterations", 10000)
        sampleGoalProbability: float = config.get("sampleGoalProbability", 0.0)
        orthogonalityMargin: float = config.get("orthogonalityMargin", 0.0)
        collisionDetectionSteps = config.get("collisionDetectionSteps", 40)
        numberOfGeneratedNodes = config.get("numberOfGeneratedNodes", 200)
        testGoalAfterNumberOfNodes = config.get(
            "testGoalAfterNumberOfNodes", 10
        )

        while self.lastGeneratedNodeNumber < numberOfGeneratedNodes:
            if numIterations >= maxIterations:
                return [], "max_iterations_reached"
            numIterations += 1

            if (
                self.lastGeneratedNodeNumber // testGoalAfterNumberOfNodes
                > numGoalTests
            ):
                numGoalTests += 1
                proj_points: List[ProjectedPoint] = []

                edges = self.graph.edges()
                for u, v in edges:
                    u_pos = self.graph.nodes[u]["pos"]
                    v_pos = self.graph.nodes[v]["pos"]
                    proj = projectPointOnEdge(
                        self.goal_pos, u_pos, v_pos, orthogonalityMargin
                    )
                    p = ProjectedPoint(
                        point=proj[0],
                        t=proj[1],
                        edge_start_id=u,
                        edge_end_id=v,
                    )
                    proj_points.append(p)
                pos_list = [x.point for x in proj_points]
                kd_tree = KDTree(pos_list)
                _, nearest_index_in_projection_points = kd_tree.query(
                    self.goal_pos, k=1
                )
                nearest_proj_point = proj_points[
                    nearest_index_in_projection_points
                ]

                # optional: step toward goal instead of directly connecting
                #
                # candidate_step = self.stepTowardCandidate(
                #    self.goal_pos, nearest_proj_point.point, stepSize
                # )

                # check if the goal is reachable from the candidate
                if not self.collisionChecker.lineInCollision(
                    nearest_proj_point.point,
                    self.goal_pos,
                    steps=collisionDetectionSteps,
                ):
                    goal_node_id = self.createNode(self.goal_pos, type="goal")

                    if (
                        nearest_proj_point.t > orthogonalityMargin
                        and nearest_proj_point.t < 1 - orthogonalityMargin
                    ):
                        new_node_id = self.divide_edge(
                            nearest_proj_point.point,
                            nearest_proj_point.edge_start_id,
                            nearest_proj_point.edge_end_id,
                        )
                        self.graph.add_edge(new_node_id, goal_node_id)
                        # Check if graph is still acyclic and connected
                        if not nx.is_tree(self.graph):
                            raise Exception(
                                "The graph is either not connected or not acyclic",
                                self.graph.edges,
                                self.graph.nodes,
                            )
                    elif nearest_proj_point.t <= orthogonalityMargin:
                        self.graph.add_edge(
                            nearest_proj_point.edge_start_id,
                            goal_node_id,
                        )
                    else:
                        self.graph.add_edge(
                            nearest_proj_point.edge_end_id,
                            goal_node_id,
                        )
                    mapping = {0: "start", goal_node_id: "goal"}
                    self.graph = nx.relabel_nodes(self.graph, mapping)
                    return self.getShortestPathFromStartToGoal(), None

            random_free_pos = self.sampleNextPosition(sampleGoalProbability)

            edges = self.graph.edges()

            # handle case when there are no edges yet (only start node)
            if not edges:
                candidate_step = self.stepTowardCandidate(
                    random_free_pos, self.start_pos, stepSize
                )
                if not self.collisionChecker.lineInCollision(
                    self.start_pos,
                    candidate_step,
                    steps=collisionDetectionSteps,
                ):
                    candidate_node_id = self.createNode(candidate_step)
                    self.graph.add_edge(0, candidate_node_id)
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
            pos_list = [x.point for x in proj_points]
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
                candidate_node_id = self.createNode(candidate_step)

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
                    if not nx.is_tree(self.graph):
                        raise Exception(
                            "The graph is either not connected or not acyclic",
                            self.graph.edges,
                            self.graph.nodes,
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

        return [], "max_nodes_reached"
