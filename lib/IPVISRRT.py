# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from lib.IPPRMBase import PRMBase


def rrtPRMVisualize(
    planner: PRMBase,
    solution: List[np.ndarray],
    ax: plt.Axes | None = None,
    nodeSize: int = 300,
):
    """Draw graph, obstacles and solution in a axis environment of matplotib."""
    graph = planner.graph
    collChecker = planner.collisionChecker
    pos_map = nx.get_node_attributes(graph, "pos")  # type: ignore
    # draw graph

    collChecker.drawObstacles(ax)
    nx.draw_networkx_nodes(
        graph, pos_map, ax=ax, cmap=cm.Blues, node_size=nodeSize
    )
    nx.draw_networkx_edges(graph, pos_map, ax=ax)

    # draw nodes based on solution path
    Gsp = nx.subgraph(graph, solution)  # type: ignore
    nx.draw_networkx_nodes(
        Gsp, pos_map, node_size=nodeSize, node_color="g", ax=ax  # type: ignore
    )

    # draw edges based on solution path
    nx.draw_networkx_edges(
        Gsp, pos_map, alpha=0.8, edge_color="g", width=3.0, ax=ax  # type: ignore
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
