# coding: utf-8
"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein). It is based on the slides/notebooks given during the course, so please **read this information first**

Important remark: Kudos to Gergely Soti, who did provide this code to integrate planar robots into the lecture

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import List

import numpy as np
import sympy as sp


class PlanarJoint:
    def __init__(self, a: float = 1.5, init_theta: float = 0, id: int = 0):
        self.a: float = a
        self.theta: float = init_theta
        self.sym_a, self.sym_theta = sp.symbols(f"a_{id} theta_{id}")
        self.rot: sp.Matrix = sp.Matrix(
            [
                [sp.cos(self.sym_theta), -sp.sin(self.sym_theta)],
                [sp.sin(self.sym_theta), sp.cos(self.sym_theta)],
            ]
        )
        self.trans: sp.Matrix = sp.Matrix(
            [
                self.sym_a * sp.cos(self.sym_theta),
                self.sym_a * sp.sin(self.sym_theta),
            ]
        )
        last_row = sp.Matrix([[0, 0, 1]])
        self.M: sp.Matrix = sp.Matrix.vstack(
            sp.Matrix.hstack(self.rot, self.trans), last_row
        )

    def get_subs(self) -> dict[sp.Symbol, float]:
        return {self.sym_a: self.a, self.sym_theta: self.theta}

    def get_transform(self) -> np.ndarray:
        return np.squeeze(
            np.array(self.M.subs(self.get_subs()) * sp.Matrix([0, 0, 1]))
        ).astype(np.float32)[:2]

    def move(self, new_theta: float):
        self.theta = new_theta


class PlanarRobot:
    def __init__(self, n_joints: int = 2):
        self.dim: int = n_joints
        self.joints: list[PlanarJoint] = [
            PlanarJoint(id=i) for i in range(self.dim)
        ]
        self.Ms: list[sp.Matrix] = [sp.eye(3)]
        for joint in self.joints:
            self.Ms.append(self.Ms[-1] * joint.M)

    def get_transforms(self) -> List[np.ndarray]:
        ts: List[np.ndarray] = []
        sub: dict[sp.Symbol, float] = {}
        for joint in self.joints:
            sub = {**sub, **joint.get_subs()}
        for M in self.Ms:
            ts.append(
                np.squeeze(
                    np.array(M.subs(sub) * sp.Matrix([0, 0, 1]))
                ).astype(np.float32)[:2]
            )
        return ts

    def move(self, new_thetas: np.ndarray | List[float]):
        assert len(new_thetas) == len(self.joints)
        for i in range(len(self.joints)):
            self.joints[i].move(new_thetas[i])


if __name__ == "__main__":
    j = PlanarJoint()
    print(j.get_transform())
    j.move(0.8)
    print(j.get_transform())
    j.move(0.6)
    print(j.get_transform())
    j.move(1.552)
    print(j.get_transform())

    r = PlanarRobot()
    r.move([0, 0.777])
    print(r.get_transforms())
