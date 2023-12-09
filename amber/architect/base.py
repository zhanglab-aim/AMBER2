"""Typing definition and hints for amber.architect
"""
from typing import List, Tuple, Dict, Set, Callable, Optional, Union, Sequence
from .. import backend as F
import numpy as np
from collections import OrderedDict
import copy


class BaseSearcher:
    def sample(self) -> Tuple[np.ndarray, list]:
        pass

    def evaluate(self, input_arc: Union[np.ndarray, list]) -> list:
        pass

    def train(self) -> float:
        pass


class ModelVariable:
    def __init__(self, name: str, min: Union[int, float], max: Union[int, float]):
        self.min = min
        self.max = max
        self.__name = name
        self.__value = None

    @property
    def name(self):
        return str(self.__name)

    def set(self, value):
        val = self.check(value)
        self.__value = val
    
    def get(self) -> int:
        return copy.copy(self.__value)

    def __repr__(self) -> str:
        return f'Variable({str(self.get())}'


class IntegerModelVariable(ModelVariable):
    def check(self, value) -> int:
        assert type(value) in (int, float), TypeError('must be a number')
        assert int(value) == value, ValueError('has fractional numbers')
        assert self.min <= value <= self.max, ValueError('out of range')
        return int(value)


class ContinuousModelVariable(ModelVariable):
    def check(self, value):
        assert self.min <= value <= self.max, ValueError('out of range')
        val = float(value)
        return val


class BaseSearchSpace(OrderedDict):
    def add_state(self, state: ModelVariable):
        self[state.name] = state
    
    def delete_state(self, state_id: str):
        del self[state_id]
    
    @staticmethod
    def from_list(d):
        assert type(d) is list
        ss = BaseSearchSpace()
        for i in range(len(d)):
            ss.add_state(state=d[i])
        return ss