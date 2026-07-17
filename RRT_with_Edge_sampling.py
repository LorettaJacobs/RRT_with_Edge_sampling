from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from lib.IPEnvironment import CollisionChecker
from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPRMBase import PRMBase
from PointProjection import projectPointOnEdge


class configDict(TypedDict):
    numberOfGeneratedNodes: int
    testGoalAfterNumberOfNodes: int
    stepSize: float
    balanceTree: bool
    sampleGoalProbability: float
    collisionDetectionSteps: int
    maxIterations: int


class ProjectedPoint(NamedTuple):
    point: np.ndarray
    t: float
    edge_start_id: int
    edge_end_id: int


type Point = np.ndarray


class RRTEdge(PRMBase):

    def __init__(self, collChecker: CollisionChecker):
        """
        _collChecker: the collision checker interface
        """
        super(RRTEdge, self).__init__(collChecker)
        self.lastGeneratedNodeNumber: int = 0

    def divide_edge(self, edgePoint: Point, start_Id: int, end_Id: int):
        # Remove old edge
        print("removing edge", start_Id, end_Id)
        self.graph.remove_edge(start_Id, end_Id)

        # Add new edges and point
        print(
            "adding node",
            self.lastGeneratedNodeNumber,
            "at position",
            edgePoint,
        )
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=edgePoint)

        self.graph.add_edge(start_Id, self.lastGeneratedNodeNumber)
        self.graph.add_edge(self.lastGeneratedNodeNumber, end_Id)

        self.lastGeneratedNodeNumber += 1

        # Check if graph is still acyclic and connected
        if not nx.is_tree(self.graph):
            raise Exception("The graph is either not connected or not acyclic")

        return self.lastGeneratedNodeNumber - 1

    def sampleNextPosition(
        self,
        sampleGoalProbability: float,
    ) -> np.ndarray:
        if sampleGoalProbability > np.random.rand():
            goalNode = self.graph.nodes[1]
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
        config: configDict,
    ) -> Tuple[List[np.ndarray], Optional[str]]:
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:
            config["numberOfGeneratedNodes"] = 500
            config["testGoalAfterNumberOfNodes"]  = 10
            config["balanceTree"] = True
            config["stepSize"] = 1.0
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
        )
        self.lastGeneratedNodeNumber += 1

        stepSize = config.get("stepSize", np.inf)

        numIterations: int = 0
        maxIterations: int = config.get("maxIterations", 10000)
        sampleGoalProbability: float = config.get("sampleGoalProbability", 0.0)

        collisionDetectionSteps = config.get("collisionDetectionSteps", 40)
        while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:
            if numIterations >= maxIterations:
                return [], "max_iterations_reached"
            numIterations += 1

            if (
                self.lastGeneratedNodeNumber
                % config["testGoalAfterNumberOfNodes"]
            ) == 0:
                proj_points: List[ProjectedPoint] = []

                edges = self.graph.edges()
                for u, v in edges:
                    u_pos = self.graph.nodes[u]["pos"]
                    v_pos = self.graph.nodes[v]["pos"]
                    proj = projectPointOnEdge(checkedGoalList[0], u_pos, v_pos)
                    p = ProjectedPoint(
                        point=proj[0],
                        t=proj[1],
                        edge_start_id=u,
                        edge_end_id=v,
                    )
                    proj_points.append(p)
                pos_list = [x.point for x in proj_points]
                print("test goal pos_list", pos_list)
                kd_tree = KDTree(pos_list)
                _, nearest_index_in_projection_points = kd_tree.query(
                    checkedGoalList[0], k=1
                )
                nearest_proj_point = proj_points[
                    nearest_index_in_projection_points
                ]
                if not self.collisionChecker.lineInCollision(
                    nearest_proj_point.point,
                    checkedGoalList[0],
                    steps=collisionDetectionSteps,
                ):
                    self.graph.add_node(
                        self.lastGeneratedNodeNumber,
                        pos=checkedGoalList[0],
                    )
                    self.lastGeneratedNodeNumber += 1

                    if nearest_proj_point.t > 0 and nearest_proj_point.t < 1:
                        new_node_id = self.divide_edge(
                            nearest_proj_point.point,
                            nearest_proj_point.edge_start_id,
                            nearest_proj_point.edge_end_id,
                        )
                        self.graph.add_edge(new_node_id, "goal")
                    elif nearest_proj_point.t <= 0:
                        self.graph.add_edge(
                            nearest_proj_point.edge_start_id,
                            "goal",
                        )
                    else:
                        self.graph.add_edge(
                            nearest_proj_point.edge_end_id,
                            "goal",
                        )
                    mapping = {0: "start"}
                    self.graph = nx.relabel_nodes(self.graph, mapping)
                    return self.getShortestPathFromStartToGoal(), None

            random_free_pos = self.sampleNextPosition(sampleGoalProbability)

            edges = self.graph.edges()

            if not edges:
                start = self.graph.nodes[0]["pos"]
                candidate_step = self.stepTowardCandidate(
                    random_free_pos, start, stepSize
                )
                if not self.collisionChecker.lineInCollision(
                    start, candidate_step, steps=collisionDetectionSteps
                ):
                    self.graph.add_node(
                        self.lastGeneratedNodeNumber,
                        pos=candidate_step,
                    )
                    self.graph.add_edge(0, self.lastGeneratedNodeNumber)
                    self.lastGeneratedNodeNumber += 1
                continue

            proj_points: List[ProjectedPoint] = []
            for u, v in edges:
                u_pos = self.graph.nodes[u]["pos"]
                v_pos = self.graph.nodes[v]["pos"]
                proj = projectPointOnEdge(random_free_pos, u_pos, v_pos)
                p = ProjectedPoint(
                    point=proj[0], t=proj[1], edge_start_id=u, edge_end_id=v
                )
                proj_points.append(p)
            pos_list = [x.point for x in proj_points]
            print("pos_list", pos_list)
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
                )
                candidate_node_id = self.lastGeneratedNodeNumber
                self.lastGeneratedNodeNumber += 1

                if nearest_proj_point.t > 0 and nearest_proj_point.t < 1:
                    new_node_id = self.divide_edge(
                        nearest_proj_point.point,
                        nearest_proj_point.edge_start_id,
                        nearest_proj_point.edge_end_id,
                    )
                    self.graph.add_edge(new_node_id, candidate_node_id)
                elif nearest_proj_point.t <= 0:
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
