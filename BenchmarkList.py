""" 
Author: Loretta Jacobs
Some parts were copied or inspired by IPTestSuite.py
"""

from typing import Dict, List

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from lib.IPBenchmark import Benchmark
from lib.IPEnvironment import CollisionChecker

benchList: List[Benchmark] = list()

# Open field
openField: Dict[str, BaseGeometry] = dict()
description = "This tests the basic functionality of if a path can be found."
benchList.append(
    Benchmark(
        "Open Field",
        CollisionChecker(openField),
        [[10, 15]],
        [[10, 1]],
        description,
        1,
    )
)

# Gate or narrow bottleneck example
gate: Dict[str, BaseGeometry] = dict()
gate["obs1"] = LineString([(0, 12), (10, 12)]).buffer(.5)
gate["obs2"] = LineString([(13, 12), (23, 12)]).buffer(.5)

description = "Planner has to go through a gate."
benchList.append( 
    Benchmark(
        "Gate",
        CollisionChecker(gate),
        [[4, 21]],
        [[7, 3]],
        description,
        2,
    )
)

# fatBottleNeckField (copied from IPTestSuite)
fatBottleNeckField: Dict[str, BaseGeometry] = dict()
fatBottleNeckField["obs1"] = Polygon(
    [(0, 8), (11, 8), (11, 15), (0, 15)]
).buffer(0.5)
fatBottleNeckField["obs2"] = Polygon(
    [(13, 8), (24, 8), (24, 15), (13, 15)]
).buffer(0.5)
description = "Planer has to find a narrow passage with a significant extend."
benchList.append(
    Benchmark(
        "Fat bottleneck",
        CollisionChecker(fatBottleNeckField),
        [[4, 21]],
        [[18, 1]],
        description,
        2,
    )
)

# Snake-like Path
snakePath: Dict[str, BaseGeometry] = dict()
snakePath["obsLeft"] = LineString([(8, 18), (23, 18), (23, 8), (8,8)]).buffer(.5)
snakePath["obsRight"] = LineString([(18, 13), (0, 13), (0, 3), (18,3)]).buffer(.5)

description = "Planner has to find an s-shaped path."
benchList.append( 
    Benchmark(
        "Snake Path",
        CollisionChecker(snakePath),
        [[4, 21]],
        [[20, 3]],
        description,
        3,
    )
)