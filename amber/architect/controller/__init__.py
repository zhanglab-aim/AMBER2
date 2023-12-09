# -*- coding: UTF-8 -*-

"""
Implementations of Recurrent RL controller for searching architectures
"""

# Changelog
#----------
#    - Aug. 7, 2018: initial
#    - Feb. 6. 2019: finished initial OperationController
#    - Jun. 17. 2019: separated to OperationController and GeneralController
#    - Aug. 15, 2020: updated documentations
#    - Dec. 10, 2022: support multi-backend
#    - Dec. 8, 2023: move GeneralController to RecurrentRLController

import sys
from .base_controller import BaseController
from ...backend import mod_name, _gen_missing_api


controller_cls = ['RecurrentRLController',]


if mod_name == 'tensorflow_1':
    from . import tf1_impl as mod
elif mod_name == 'tensorflow_2':
    from . import tf2_impl as mod
elif mod_name == 'pytorch':
    from . import torch_impl as mod
else:
    raise Exception(f"Unsupported {mod_name} backend for controller")

thismod = sys.modules[__name__]
for api in controller_cls:
    if api in mod.__dict__:
        setattr(thismod, api, mod.__dict__[api])
    else:
        setattr(thismod, api, _gen_missing_api(api, mod_name))


__all__ = [
    'BaseController',
] + controller_cls

