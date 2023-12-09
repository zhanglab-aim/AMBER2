"""
General controller for searching computational operation per layer, and residual connection
"""

import os
import sys
import numpy as np
import h5py
from ... import backend as F
from .base_controller import BaseController, proximal_policy_optimization_loss
from .base_controller import get_kl_divergence_n_entropy, convert_arc_to_onehot
import tensorflow as tf


class RecurrentRLController(BaseController):
    def train_step(self, input_arc, advantage, old_probs):
        with tf.GradientTape() as tape:
            advantage = F.cast(advantage, F.float32)
            old_probs = [F.cast(p, F.float32) for p in old_probs]
            onehot_log_prob, probs, skip_count, onehot_skip_penaltys = self.evaluate(input_arc=input_arc)
            #normalize = F.cast(self.num_layers * (self.num_layers - 1) / 2, F.float32)
            #self.skip_rate = F.cast(self.skip_count, F.float32) / normalize
            loss = 0
            if self.skip_weight is not None:
                loss += self.skip_weight * F.reduce_mean(onehot_skip_penaltys)
            if self.use_ppo_loss:
                raise NotImplementedError(f"No PPO support for {F.mod_name} yet")
            else:
                loss += F.reshape(F.tensordot(onehot_log_prob, advantage, axes=1), [])

            self.input_arc = input_arc
            input_arc_onehot = convert_arc_to_onehot(self)
            kl_div, ent = get_kl_divergence_n_entropy(curr_prediction=probs,
                                                                old_prediction=old_probs,
                                                                curr_onehot=input_arc_onehot,
                                                                old_onehotpred=input_arc_onehot)
        F.get_train_op(
            loss=loss,
            variables=self.params,
            optimizer=self.optimizer,
            tape=tape
        )
        return loss, kl_div, ent

