"""Main entry for AMBER"""

from . import backend
from . import architect, modeler
from . import utils
#from .wrapper import Amber


__version__ = "2.0.0b0"

__all__ = [
    'backend',
    'architect',
    'modeler',
    'utils'
]
