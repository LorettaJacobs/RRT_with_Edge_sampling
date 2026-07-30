"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

Author: Gergely Soti, adapted by Bjoern Hein

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from shapely import plotting
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from lib.IPEnvironment import CollisionChecker
from lib.IPPerfMonitor import IPPerfMonitor
from lib.IPPlanarManipulator import PlanarRobot
from lib.IPRRT import RRT


def interpolate_line(
    startPos: np.ndarray,
    endPos: np.ndarray,
    step_l: float,
) -> List[np.ndarray]:
    steps: List[np.ndarray] = []
    line = np.array(endPos) - np.array(startPos)
    line_l = np.linalg.norm(line)
    step = line / line_l * step_l
    n_steps = np.floor(line_l / step_l).astype(np.int32)
    c_step = np.array(startPos)
    for _ in range(n_steps):
        steps.append(copy.deepcopy(c_step))
        c_step += step
    if not (c_step == np.array(endPos)).all():
        steps.append(np.array(endPos))
    return steps


class KinChainCollisionChecker(CollisionChecker):
    def __init__(
        self,
        kin_chain: PlanarRobot,
        scene: Dict[str, BaseGeometry],
        limits: List[List[float]] | None = [[-3.0, 3.0], [-3.0, 3.0]],
        statistic: Any = None,
        fk_resolution: float = 0.1,
    ):
        super(KinChainCollisionChecker, self).__init__(
            scene, limits, statistic
        )
        if len(self.limits) != kin_chain.dim:
            raise ValueError(
                "Limits must match the dimension of the kinematic chain. Default values are for a 2-dof planar manipulator. If you use dof>2 you have to specify the limits explicitly"
            )
        self.kin_chain = kin_chain
        self.fk_resolution = fk_resolution
        self.dim = self.kin_chain.dim

    def getDim(self):
        return self.dim

    def pointInCollision(self, pos: np.ndarray) -> bool:
        self.kin_chain.move(pos)
        joint_positions = self.kin_chain.get_transforms()
        for i in range(1, len(joint_positions)):
            if self.segmentInCollision(
                joint_positions[i - 1], joint_positions[i]
            ):
                return True
        return False

    @IPPerfMonitor
    def lineInCollision(
        self,
        startPos: np.ndarray,
        endPos: np.ndarray,
        steps: int | None = None,
    ) -> bool:
        assert len(startPos) == self.getDim()
        assert len(endPos) == self.getDim()
        interpolate_steps = interpolate_line(
            startPos, endPos, self.fk_resolution
        )
        for pos in interpolate_steps:
            if self.pointInCollision(pos):
                return True
        return False

    def segmentInCollision(
        self, startPos: np.ndarray, endPos: np.ndarray
    ) -> bool:
        for _, value in self.scene.items():
            if value.intersects(
                LineString(
                    [(startPos[0], startPos[1]), (endPos[0], endPos[1])]
                )
            ):
                return True
        return False

    def drawObstacles(self, ax: plt.Axes | None, inWorkspace: bool = False):
        if inWorkspace:
            for _, value in self.scene.items():
                plotting.plot_polygon(
                    value, add_points=False, color="red", ax=ax
                )


def planarRobotVisualize(
    kin_chain: PlanarRobot, ax: plt.Axes, color: str = "g"
):
    joint_positions = kin_chain.get_transforms()
    for i in range(1, len(joint_positions)):
        xs = [joint_positions[i - 1][0], joint_positions[i][0]]
        ys = [joint_positions[i - 1][1], joint_positions[i][1]]
        ax.plot(xs, ys, color=color)


matplotlib.rcParams["animation.embed_limit"] = 64


def animateSolution(
    planner: RRT,
    environment: KinChainCollisionChecker,
    solution: List[np.ndarray],
    visualizer: Callable[[RRT, List[np.ndarray], plt.Axes, int | None], None],
    workSpaceLimits: List[Tuple[float, float]] = [(-3, 3), (-3, 3)],
    path: str | None = None,
):
    _planner = planner
    _environment = environment
    _solution = solution
    _prmVisualizer = visualizer

    if _environment.getDim() == 2:

        fig_local = plt.figure(figsize=(14, 7))
        ax1 = fig_local.add_subplot(1, 2, 1)
        ax2 = fig_local.add_subplot(1, 2, 2)
        # get positions for solution
        solution_pos = [
            _planner.graph.nodes[node]["pos"] for node in _solution
        ]
        # interpolate to obtain a smoother movement
        i_solution_pos = [solution_pos[0]]
        for i in range(1, len(solution_pos)):
            segment_s = solution_pos[i - 1]
            segment_e = solution_pos[i]
            i_solution_pos = (
                i_solution_pos
                + interpolate_line(segment_s, segment_e, 0.1)[1:]
            )
        # animate
        frames = len(i_solution_pos)

        r = environment.kin_chain

        def animate(t: int):
            # clear task space figure
            ax1.cla()
            # fix figure size
            ax1.set_xlim(*workSpaceLimits[0])
            ax1.set_ylim(*workSpaceLimits[1])
            # draw obstacles
            _environment.drawObstacles(ax1, inWorkspace=True)
            # update robot position
            pos = i_solution_pos[t]
            r.move(pos)
            planarRobotVisualize(r, ax1)

            # clear joint space figure
            ax2.cla()
            # draw graph and path
            _prmVisualizer(_planner, solution, ax2, None)
            # draw current position in joint space
            ax2.scatter(
                i_solution_pos[t][0],
                i_solution_pos[t][1],
                color="r",
                zorder=10,
                s=250,
            )

        ani = matplotlib.animation.FuncAnimation(
            fig_local, animate, frames=frames
        )
        html = HTML(ani.to_jshtml())
        if path is not None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            path = f"{path}/animation_{_planner.__class__.__name__}.html"
            with open(path, "w") as f:
                f.write(ani.to_jshtml())
        display(html)
        plt.close()
    else:
        fig_local = plt.figure(figsize=(7, 7))
        ax1 = fig_local.add_subplot(1, 1, 1)
        # get positions for solution
        solution_pos = [
            _planner.graph.nodes[node]["pos"] for node in _solution
        ]
        # interpolate to obtain a smoother movement
        i_solution_pos = [solution_pos[0]]
        for i in range(1, len(solution_pos)):
            segment_s = solution_pos[i - 1]
            segment_e = solution_pos[i]
            i_solution_pos = (
                i_solution_pos
                + interpolate_line(segment_s, segment_e, 0.1)[1:]
            )
        # animate
        frames = len(i_solution_pos)

        r = environment.kin_chain

        def animate(t: int):
            # clear task space figure
            ax1.cla()
            # fix figure size
            ax1.set_xlim(*workSpaceLimits[0])
            ax1.set_ylim(*workSpaceLimits[1])
            # draw obstacles
            _environment.drawObstacles(ax1, inWorkspace=True)
            # update robot position
            pos = i_solution_pos[t]
            r.move(pos)
            planarRobotVisualize(r, ax1)

        ani = matplotlib.animation.FuncAnimation(
            fig_local, animate, frames=frames
        )
        html = HTML(ani.to_jshtml())
        if path is not None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            path = f"{path}/animation_{_planner.__class__.__name__}.html"
            with open(path, "w") as f:
                f.write(ani.to_jshtml())
        display(html)
        plt.close()
