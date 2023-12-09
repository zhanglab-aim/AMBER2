from typing import Tuple, List, Union, Dict, Any, Optional, Mapping
from argparse import Namespace
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional


class ExperimentWriter:
    """In-memory experiment writer.
    Currently this supports logging hyperparameters and metrics.
    
    Borrowing from @ttesileanu https://github.com/ttesileanu/cancer-net
    """

    def __init__(self):
        self.hparams: Dict[str, Any] = {}
        self.metrics: List[Dict[str, float]] = []

    def log_hparams(self, params: Dict[str, Any]):
        """Record hyperparameters.
        This adds to previously recorded hparams, overwriting exisiting values in case
        of repeated keys.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Record metrics."""

        def _handle_value(value: Union[torch.Tensor, Any]) -> Any:
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)


