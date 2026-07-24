""" 
Autor: Loretta Jacobs
"""

import unittest
import networkx as nx
from Divide_Edge import divide_edge 

class TestDivideEdge(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.lastGeneratedNodeNumber = 0

    def test_divided_Edge(self):
        
        start = [0, 0]
        startId = 0
        end = [1, 0]
        endId = 1
        point = [0.5, 0]        
        self.lastGeneratedNodeNumber += 2

        self.graph.add_node(startId, pos=start)
        self.graph.add_node(endId, pos=end)
        self.graph.add_edge(startId, endId)
        
        pointId = divide_edge(self, point, startId, endId)

        self.assertEqual(len(self.graph.nodes), 3)
        self.assertTrue(self.graph.has_edge(startId, pointId))
        self.assertTrue(self.graph.has_edge(pointId, endId))

        self.assertTrue(nx.is_tree(self.graph))

    def test_divided_Edge_moreNodes(self):
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=[0,0])
        firstPointId = self.lastGeneratedNodeNumber
        self.lastGeneratedNodeNumber += 1

        self.graph.add_node(self.lastGeneratedNodeNumber, pos=[5,0])
        secondPointId = self.lastGeneratedNodeNumber
        self.lastGeneratedNodeNumber += 1

        self.graph.add_edge(firstPointId, secondPointId)

        self.graph.add_node(self.lastGeneratedNodeNumber, pos=[0,3])
        thirdPointId = self.lastGeneratedNodeNumber
        self.lastGeneratedNodeNumber += 1

        self.graph.add_edge(firstPointId, thirdPointId)
        
       
        start = [5,3]
        startId = self.lastGeneratedNodeNumber
        self.lastGeneratedNodeNumber += 1
        end = [7, 6]
        endId = self.lastGeneratedNodeNumber
        self.lastGeneratedNodeNumber += 1

        self.graph.add_node(startId, pos=start)
        self.graph.add_node(endId, pos=end)
        self.graph.add_edge(secondPointId, startId)
        self.graph.add_edge(startId, endId)

        point = [6, 4]   
        pointId = divide_edge(self, point, startId, endId)

        self.assertEqual(len(self.graph.nodes), 6)
        self.assertTrue(self.graph.has_edge(startId, pointId))
        self.assertTrue(self.graph.has_edge(pointId, endId))

        self.assertTrue(nx.is_tree(self.graph))

    def tearDown(self):
        self.graph.clear()
