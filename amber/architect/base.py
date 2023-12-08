"""Typing definition and hints for amber.architect
"""
from typing import List, Tuple, Dict, Set, Callable, Optional, Union, Sequence
from amber import backend as F
import numpy as np


class BaseSearcher:
    pass


class BaseModelSpace:
    pass


class BaseReward:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model: F.Model, data: Union[np.ndarray, Tuple, List], *args, **kwargs) -> Tuple[float, list, list]:
        return 0, [0,], []