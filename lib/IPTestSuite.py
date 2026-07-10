# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).
It gathers all visualizations of the investigated and explained planning algorithms.
License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Dict, List

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from lib.IPBenchmark import Benchmark
from lib.IPEnvironment import CollisionChecker

benchList: List[Benchmark] = list()

# -----------------------------------------
trapField: Dict[str, BaseGeometry] = dict()
trapField["obs1"] = LineString([(6, 18), (6, 8), (16, 8), (16, 18)]).buffer(
    1.0
)
description = "Following the direct connection from goal to start would lead the algorithm into a trap."
benchList.append(
    Benchmark(
        "Trap",
        CollisionChecker(trapField),
        [[10, 15]],
        [[10, 1]],
        description,
        2,
    )
)

# -----------------------------------------
bottleNeckField: Dict[str, BaseGeometry] = dict()
bottleNeckField["obs1"] = LineString([(0, 13), (11, 13)]).buffer(0.5)
bottleNeckField["obs2"] = LineString([(13, 13), (23, 13)]).buffer(0.5)
description = "Planer has to find a narrow passage."
benchList.append(
    Benchmark(
        "Bottleneck",
        CollisionChecker(bottleNeckField),
        [[4, 15]],
        [[18, 1]],
        description,
        2,
    )
)

# -----------------------------------------
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

# -----------------------------------------

myField: Dict[str, BaseGeometry] = dict()
myField["L"] = Polygon(
    [(10, 16), (10, 11), (13, 11), (13, 12), (11, 12), (11, 16)]
)
myField["T"] = Polygon(
    [
        (14, 16),
        (14, 15),
        (15, 15),
        (15, 11),
        (16, 11),
        (16, 15),
        (17, 15),
        (17, 16),
    ]
)
myField["C"] = Polygon(
    [
        (19, 16),
        (19, 11),
        (22, 11),
        (22, 12),
        (20, 12),
        (20, 15),
        (22, 15),
        (22, 16),
    ]
)

myField["Antenna_L"] = Polygon([(3, 12), (1, 16), (2, 16), (4, 12)])
myField["Antenna_Head_L"] = Point(1.5, 16).buffer(1)

myField["Antenna_R"] = Polygon([(7, 12), (9, 16), (8, 16), (6, 12)])
myField["Antenna_Head_R"] = Point(8.5, 16).buffer(1)

myField["Rob_Head"] = Polygon([(2, 13), (2, 8), (8, 8), (8, 13)])
description = (
    "Planer has to find a passage past a robot head and the print of the LTC."
)
benchList.append(
    Benchmark(
        "MyField",
        CollisionChecker(myField),
        [[4, 21]],
        [[18, 1]],
        description,
        2,
    )
)
