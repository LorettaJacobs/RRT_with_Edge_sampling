""" 
Author: Loretta Jacobs
"""

import networkx as nx
import numpy as np

type Point = np.ndarray

def divide_edge(self, edgePoint: Point, start_Id: int, end_Id: int) -> int:
        # Remove old edge
        self.graph.remove_edge(start_Id, end_Id)

        # Add new edges and point
        newNodeId = self.lastGeneratedNodeNumber
        self.graph.add_node(newNodeId, pos=edgePoint)
        self.graph.add_edge(start_Id, newNodeId)
        self.graph.add_edge(newNodeId, end_Id)
        self.lastGeneratedNodeNumber += 1
        
        # Check if graph is still acyclic and connected
        if(not nx.is_tree(self.graph)):
            raise Exception(
                "The graph is either not connected or not acyclic"
            )
        
        return newNodeId
        
        