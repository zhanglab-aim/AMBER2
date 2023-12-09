"""
General controller for searching computational operation per layer, and residual connection
"""

import h5py
import numpy as np
import os
from .base_controller import BaseController, proximal_policy_optimization_loss
from .base_controller import get_kl_divergence_n_entropy


# dynamic graphs
class RecurrentRLController(BaseController):
    def save_weights(self, filepath):
        #weights = self.get_weights()
        with h5py.File(filepath, "w") as hf:
            pass
            #for i, d in enumerate(weights):
            #    hf.create_dataset(name=self.weights[i].name, data=d)

    def load_weights(self, *args):
        pass
