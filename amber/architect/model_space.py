# -*- coding: UTF-8 -*-

"""
Model space to perform architecture search
"""

# Author       : ZZJ
# Initial Date : Nov. 17, 2018
# Last Update  : Aug. 16, 2020

from collections import defaultdict, OrderedDict
import copy
import numpy as np
from typing import Union, Optional
import numpy as np
from ..backend import Operation
from .base import BaseSearchSpace


class ModelVariable:
    def __init__(self, name: str, min: Union[int, float], max: Union[int, float], init_value: Optional[Union[int, float]]=None):
        self.min = min
        self.max = max
        self.name = name
        self.value = init_value
    
    def uniform_sample(self):
        return np.random.uniform(self.min, self.max)

    #@property
    #def value(self):
    #    return self.get()

    def set(self, value):
        val = self.check(value)
        self.value = val
    
    def get(self) -> int:
        return copy.copy(self.value)

    def __repr__(self) -> str:
        return f'Variable(name={self.name}, value={str(self.value)})'


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


class ModelSpace(OrderedDict, BaseSearchSpace):
    def __getitem__(self, var_id: Union[str, int]):
        if type(var_id) is str:
            return self.get(var_id)
        elif type(var_id) is int:
            return self[list(self.keys())[var_id]]
        else:
            raise TypeError("var_id can only be string or int")

    def add_state(self, state: ModelVariable):
        assert state.name not in self.keys(), IndexError(f'Varialble name {state.name} already exists in space')
        self[state.name] = state
    
    def delete_state(self, state_id: str):
        del self[state_id]
    
    @staticmethod
    def from_list(d):
        assert type(d) is list
        vs = ModelSpace()
        for i in range(len(d)):
            assert isinstance(d[i], ModelVariable)
            vs.add_state(state=d[i])
        return vs


class OperationSpace(BaseSearchSpace):
    """Operation Model Space constructor

    Provides utility functions for holding "operations" that the controller must use to train and predict.
    Also provides a more convenient way to define the model search space

    There are several ways to construct a model space. For example, one way is to initialize an empty ``ModelSpace`` then
    iteratively add layers to it, where each layer has a number of candidate operations::

        >>> def get_model_space(out_filters=64, num_layers=9):
        >>>    model_space = ModelSpace()
        >>>    num_pool = 4
        >>>    expand_layers = [num_layers//num_pool*i-1 for i in range(1, num_pool)]
        >>>    for i in range(num_layers):
        >>>        model_space.add_layer(i, [
        >>>            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
        >>>            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
        >>>            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
        >>>            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
        >>>            Operation('identity', filters=out_filters),
        >>>      ])
        >>>        if i in expand_layers:
        >>>            out_filters *= 2
        >>>    return model_space

    Alternatively, ModelSpace can also be constructed from a dictionary.

    """

    def __init__(self, **kwargs):
        self.state_space = defaultdict(list)

    def __str__(self):
        return "StateSpace with {} layers and {} total combinations".format(len(self.state_space),
                                                                            self.get_space_size())

    def __len__(self):
        return len(self.state_space)

    def __getitem__(self, layer_id):
        if layer_id < 0:
            layer_id = len(self.state_space) + layer_id
        if layer_id not in self.state_space:
            raise IndexError('layer_id out of range')
        return self.state_space[layer_id]

    def __setitem__(self, layer_id, layer_states):
        self.add_layer(layer_id, layer_states)

    def add_state(self, layer_id, state):
        """Append a new state/operation to a layer

        Parameters
        ----------
        layer_id : int
            Which layer to append a new operation.

        state : amber.architect.State
            The new operation object to be appended.

        Returns
        -------

        """
        self.state_space[layer_id].append(state)

    def delete_state(self, layer_id, state_id):
        """Delete an operation from layer

        Parameters
        ----------
        layer_id : int
            Which layer to delete an operation

        state_id : int
            Which operation index to be deleted

        Returns
        -------

        """
        del self.state_space[layer_id][state_id]

    def add_layer(self, layer_id, layer_states=None):
        """Add a new layer to model space

        Parameters
        ----------
        layer_id : int
            The layer id of which layer to be added. Can be incontinuous to previous layers.

        layer_states : list of amber.architect.Operation
            A list of ``Operation`` object to be added.

        Returns
        -------
        bool
            Boolean value of Whether the model space is valid after inserting this layer
        """
        if layer_states is None:
            self.state_space[layer_id] = []
        else:
            self.state_space[layer_id] = layer_states
        return self._check_space_integrity()

    def delete_layer(self, layer_id):
        """Delete an entire layer and its associated values

        Parameters
        ----------
        layer_id : int
            which layer index to be deleted

        Returns
        -------
        bool
            Boolean value of Whether the model space is valid after inserting this layer
        """
        del self.state_space[layer_id]
        return self._check_space_integrity()

    def _check_space_integrity(self):
        return len(self.state_space) - 1 == max(self.state_space.keys())

    def print_state_space(self):
        """
        print out the model space in a nice layout (not so nice yet)
        """
        for i in range(len(self.state_space)):
            print("Layer {}".format(i))
            print("\n".join(["  " + str(x) for x in self.state_space[i]]))
            print('-' * 10)
        return

    def get_random_model_states(self):
        """Get a random combination of model operations throughout each layer

        Returns
        -------
        model_states : list
            A list of randomly sampled model operations
        """
        model_states = []
        for i in range(len(self.state_space)):
            model_states.append(np.random.choice(self.state_space[i]))
        return model_states

    @staticmethod
    def from_dict(d):
        """Static method for creating a ModelSpace from a Dictionary or List

        Parameters
        ----------
        d : dict or list
            A dictionary or list specifying candidate operations for each layer

        Returns
        -------
        amber.architect.ModelSpace
            The constructed model space from the given dict/list

        """
        import ast
        assert type(d) in (dict, list)
        num_layers = len(d)
        ms = OperationSpace()
        for i in range(num_layers):
            ms.add_layer(layer_id=i, layer_states=[
                         d[i][j] if type(d[i][j]) is Operation else Operation(**d[i][j])
                         for j in range(len(d[i]))])
        return ms



# alias
State = Operation
