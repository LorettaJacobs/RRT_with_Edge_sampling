# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from lib.IPPRMBase import PRMBase


def rrtPRMVisualize(
    planner: PRMBase,
    solution: List[np.ndarray],
    ax: plt.Axes | None = None,
    nodeSize: int = 50,
    drawLabels: bool = False,
):
    """Draw graph, obstacles and solution in a axis environment of matplotib."""
    graph = planner.graph
    collChecker = planner.collisionChecker
    pos_map = nx.get_node_attributes(graph, "pos")  # type: ignore

    if pos_map:
        xs = [p[0] for p in pos_map.values()]
        ys = [p[1] for p in pos_map.values()]
        x_mid = (min(xs) + max(xs)) / 2.0
        y_mid = (min(ys) + max(ys)) / 2.0
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        max_side = max(width, height)
        # 10 % Puffer an allen Seiten
        padding = 0.1 * max_side if max_side > 0 else 1.0
        half_side = (max_side + 2 * padding) / 2.0

        ax.set_xlim(x_mid - half_side, x_mid + half_side)
        ax.set_ylim(y_mid - half_side, y_mid + half_side)
        ax.set_aspect("equal")  # gleicher Maßstab in x und y

    # draw graph

    node_modes: List[str] = list(nx.get_node_attributes(graph, "mode", default="not_bi_rrt").values())  # type: ignore
    node_types: List[str] = list(nx.get_node_attributes(graph, "type", default="normal").values())  # type: ignore
    node_colors = [
        (
            ("#7B00FF" if mode == "backward" else "#FF00D4")
            if typ == "projected"
            else "#0091FF" if mode == "forward" else "#FF5900"
        )
        for (mode, typ) in zip(node_modes, node_types)
    ]

    collChecker.drawObstacles(ax)

    nx.draw_networkx_nodes(
        graph,
        pos_map,
        ax=ax,
        node_color=node_colors,
        node_size=nodeSize,
    )
    nx.draw_networkx_edges(graph, pos_map, ax=ax)

    if drawLabels:
        labels = {n: str(n) for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos_map, labels=labels, ax=ax)

    # draw nodes based on solution path
    Gsp = nx.subgraph(graph, solution)  # type: ignore

    # draw edges based on solution path
    nx.draw_networkx_edges(
        Gsp, pos_map, alpha=0.5, edge_color="g", width=3.0, ax=ax  # type: ignore
    )

    # draw start and goal
    if "start" in graph.nodes():
        nx.draw_networkx_nodes(
            graph,
            pos_map,
            nodelist=["start"],
            node_size=nodeSize,
            node_color="#00dd00",
            ax=ax,
        )
        nx.draw_networkx_labels(graph, pos_map, labels={"start": "S"}, ax=ax)

    if "goal" in graph.nodes():
        nx.draw_networkx_nodes(
            graph,
            pos_map,
            nodelist=["goal"],
            node_size=nodeSize,
            node_color="#DD0000",
            ax=ax,
        )
        nx.draw_networkx_labels(graph, pos_map, labels={"goal": "G"}, ax=ax)
