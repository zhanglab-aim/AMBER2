from collections import OrderedDict
import copy
import numpy as np
from typing import Union, Optional
from .base import BaseSearchSpace

class ModelVariable:
    def __init__(self, name: str, min: Union[int, float], max: Union[int, float], init_value: Optional[Union[int, float]]=None):
        self.min = min
        self.max = max
        self.__name = name
        self.__value = init_value
    
    def uniform_sample(self):
        return np.random.uniform(self.min, self.max)

    @property
    def name(self):
        return str(self.__name)
    
    @property
    def value(self):
        return self.get()

    def set(self, value):
        val = self.check(value)
        self.__value = val
    
    def get(self) -> int:
        return copy.copy(self.__value)

    def __repr__(self) -> str:
        return f'Variable(name={self.name}, value={str(self.get())})'


class IntegerModelVariable(ModelVariable):
    def __init__(self, name: str, min: Union[int, float], max: Union[int, float], init_value: Optional[Union[int, float]] = None):
        super().__init__(name, min, max, init_value)
        if self.value is None:
            self.set(self.uniform_sample())
    
    def uniform_sample(self) -> int:
        return np.random.randint(low=self.min, high=self.max+1)

    def check(self, value) -> int:
        assert type(value) in (int, float), TypeError('must be a number')
        assert int(value) == value, ValueError('has fractional numbers')
        assert self.min <= value <= self.max, ValueError('out of range')
        return int(value)

    @property
    def num_choices(self):
        return self.max - self.min + 1
    
    def __len__(self):
        return self.num_choices

class ContinuousModelVariable(ModelVariable):
    def __init__(self, name: str, min: Union[int, float], max: Union[int, float], init_value: Optional[Union[int, float]] = None):
        super().__init__(name, min, max, init_value)
        if self.value is None:
            self.set(self.uniform_sample())
    
    def check(self, value):
        assert self.min <= value <= self.max, ValueError('out of range')
        val = float(value)
        return val

    @property
    def num_choices(self):
        return np.inf

    def __repr__(self) -> str:
        return f'Variable(name={self.name}, value={str(round(self.get(),3))})'

class VariableSpace(OrderedDict, BaseSearchSpace):
    def __getitem__(self, var_id: Union[str, int]):
        if type(var_id) is str:
            return self.get(var_id)
        elif type(var_id) is int:
            return self[list(self.keys())[var_id]]
        else:
            raise TypeError("var_id can only be string or int")

    def add_state(self, state: ModelVariable):
        self[state.name] = state
    
    def delete_state(self, state_id: str):
        del self[state_id]
    
    @staticmethod
    def from_list(d):
        assert type(d) is list
        vs = VariableSpace()
        for i in range(len(d)):
            vs.add_state(state=d[i])
        return vs
