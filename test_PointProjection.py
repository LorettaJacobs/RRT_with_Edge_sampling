import unittest
from PointProjection import projectPointOnEdge
from numpy.testing import assert_array_equal

class TestPointProjection(unittest.TestCase):
    def test_projection_on_segment(self):
        result = projectPointOnEdge([0.5, 1], [0,0], [1,0])
        self.assertTrue(result[1] > 0) 
        self.assertTrue(result[1] < 1) 
        assert_array_equal(result[0], [0.5, 0.0])

    def test_projection_outside_negative_segment(self):
        result = projectPointOnEdge([0, 1], [1,0], [2,0])
        self.assertTrue(result[1] < 0)
        assert_array_equal(result[0], [1,0])

    def test_projection_outside_positive_segment(self):
        result = projectPointOnEdge([3, 1], [1,0], [2,0])
        self.assertTrue(result[1] > 0)
        assert_array_equal(result[0], [2,0])
    
    def test_projection_next_negative_segment(self):
        result = projectPointOnEdge([0, 0], [1,0], [2,0])
        self.assertTrue(result[1] < 0)
        assert_array_equal(result[0], [1,0])
    
    def test_projection_next_positive_segment(self):
        result = projectPointOnEdge([3, 0], [1,0], [2,0])
        self.assertTrue(result[1] > 0)
        assert_array_equal(result[0], [2,0])