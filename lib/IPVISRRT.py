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
):
    """Draw graph, obstacles and solution in a axis environment of matplotib."""
    graph = planner.graph
    collChecker = planner.collisionChecker
    pos_map = nx.get_node_attributes(graph, "pos")  # type: ignore
    # draw graph

    node_modes = list(nx.get_node_attributes(graph, "mode", default="not_bi_rrt").values())  # type: ignore
    node_colors = [
        ("#0091FF" if mode == "forward" else "#FF5900") for mode in node_modes
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

    # labels = {n: str(n) for n in graph.nodes()}
    # nx.draw_networkx_labels(graph, pos_map, labels=labels, ax=ax)

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
