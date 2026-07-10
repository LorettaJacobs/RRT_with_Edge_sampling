# coding: utf-8

"""
This code is part of the course 'Introduction to robot path planning' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import time
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import pandas
from typing_extensions import TypedDict


# Typ-Definitionen
class ElementType(TypedDict):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    retVal: Any
    time: float


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class IPPerfMonitor(Generic[P, R]):
    """Performance Monitor Decorator"""

    __instances: Dict[Callable[..., Any], "IPPerfMonitor[Any, Any]"] = {}

    def __init__(self, f: Callable[P, R]) -> None:
        self.__f = f
        self.data: List[ElementType] = []
        IPPerfMonitor.__instances[f] = self

        wraps(f)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Führt die überwachte Funktion aus"""
        starttime = time.time()
        ret = self.__f(*args, **kwargs)
        endtime = time.time()

        element: ElementType = {
            "args": args,
            "kwargs": kwargs,
            "retVal": ret,
            "time": endtime - starttime,
        }
        self.data.append(element)
        return ret

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Callable[P, R]: ...

    @overload
    def __get__(
        self, instance: object, owner: type[Any]
    ) -> Callable[..., R]: ...

    def __get__(
        self, instance: Optional[T], owner: type[Any]
    ) -> Union[Callable[P, R], Callable[..., R]]:
        if instance is None:
            return self.__f
        bound_method = partial(self.__call__, cast(Any, instance))
        wraps(self.__f)(bound_method)
        return bound_method

    @staticmethod
    def dataFrame() -> pandas.DataFrame:
        result: List[Dict[str, Any]] = []
        for f in IPPerfMonitor.__instances:
            for dataElement in IPPerfMonitor.__instances[f].data:
                result.append({"name": f.__name__, **dataElement})
        return pandas.DataFrame.from_records(result)

    @staticmethod
    def clearData() -> None:
        for f in IPPerfMonitor.__instances:
            IPPerfMonitor.__instances[f].data.clear()
