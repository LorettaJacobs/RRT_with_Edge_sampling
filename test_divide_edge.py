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
        self.lastGeneratedNodeNumber = 2
        pointId = self.lastGeneratedNodeNumber

        self.graph.clear()
        self.graph.add_node(startId, pos=start)
        self.graph.add_node(endId, pos=end)
        self.graph.add_edge(startId, endId)
        
        divide_edge(self, point, startId, endId)

        self.assertEqual(len(self.graph.nodes), 3)
        self.assertTrue(self.graph.has_edge(startId, pointId))
        self.assertTrue(self.graph.has_edge(pointId, endId))

        self.assertTrue(nx.is_tree(self.graph))
