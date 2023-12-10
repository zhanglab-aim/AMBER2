"""Typing definition and hints for amber.architect
"""
from typing import List, Tuple, Dict, Set, Callable, Optional, Union, Sequence
from .. import backend as F
import numpy as np
import copy


class BaseSearcher:
    def sample(self) -> Tuple[np.ndarray, list]:
        pass

    def evaluate(self, input_arc: Union[np.ndarray, list]) -> list:
        pass

    def train(self) -> float:
        pass

class BaseSearchSpace:
    pass