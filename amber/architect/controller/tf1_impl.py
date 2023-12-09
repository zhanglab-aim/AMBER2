"""
General controller for searching computational operation per layer, and residual connection
"""

# Author       : zzjfrank
# Last Update  : Aug. 16, 2020

import os
import sys
import numpy as np
import h5py
from ... import backend as F
from .base_controller import BaseController, proximal_policy_optimization_loss
from .base_controller import get_kl_divergence_n_entropy, convert_arc_to_onehot



class RecurrentRLController(BaseController):
    def create_learner(self):
        self.loss = 0    
        with F.device_scope("/cpu:0"):
            with F.variable_scope(self.name):
                self._create_weight()
                # for static libs, need to build all tensors at once
                self._build_sampler()
                self._build_trainer()
                self.train_step()
        # initialize variables in this scope
        self.params = [var for var in F.trainable_variables(scope=self.name)]
        F.init_all_params(sess=self.session)
    
    def _build_sampler(self):
        arc_seq, probs_, log_probs, hidden_states, entropys, skip_count, skip_penaltys = self.forward(input_arc=None)
        self.sample_hidden_states = hidden_states
        # for class attr.
        arc_seq = F.concat(arc_seq, axis=0)
        self.sample_arc = F.reshape(arc_seq, [-1])
        entropys = F.stack(entropys)
        self.sample_entropy = F.reduce_sum(entropys)
        log_probs = F.stack(log_probs)
        self.sample_log_prob = F.reduce_sum(log_probs)
        if self.with_skip_connection:
            skip_count = F.stack(skip_count)
            self.skip_count = F.reduce_sum(skip_count)
        self.sample_probs = probs_
    
    def _build_trainer(self):
        ops_each_layer = 1
        total_arc_len = sum(
            [ops_each_layer] +   # first layer
            [ops_each_layer + i * self.with_skip_connection for i in range(1, self.num_layers)]  # rest layers
        )
        self.total_arc_len = total_arc_len
        self.input_arc = [F.placeholder(shape=(None, 1), dtype=F.int32, name='arc_{}'.format(i))
                          for i in range(total_arc_len)]
        arc_seq, probs_, log_probs, hidden_states, entropys, skip_count, skip_penaltys = self.forward(input_arc=self.input_arc)
        self.train_hidden_states = hidden_states

        # for class attributes
        self.entropys = F.stack(entropys)
        self.onehot_probs = probs_
        log_probs = F.stack(log_probs)
        self.onehot_log_prob = F.reshape(F.reduce_sum(log_probs, axis=0), [-1]) # (batch_size,)
        skip_count = F.stack(skip_count)
        self.onehot_skip_count = F.reduce_sum(skip_count, axis=0)
        skip_penaltys_flat = [F.reduce_mean(x, axis=1) for x in skip_penaltys] # from (num_layer-1, batch_size, layer_id) to (num_layer-1, batch_size); layer_id makes each tensor of varying lengths in the list
        self.onehot_skip_penaltys = F.reduce_mean(skip_penaltys_flat, axis=0)  # (batch_size,)

    def train_step(self):
        """build train_op based on either REINFORCE or PPO
        """
        self.advantage = F.placeholder(shape=(None, 1), dtype=F.float32, name="advantage")
        self.reward = F.placeholder(shape=(None, 1), dtype=F.float32, name="reward")

        normalize = F.cast(self.num_layers * (self.num_layers - 1) / 2, F.float32)
        self.skip_rate = F.cast(self.skip_count, F.float32) / normalize

        self.input_arc_onehot = convert_arc_to_onehot(self)
        self.old_probs = [F.placeholder(shape=self.onehot_probs[i].shape, dtype=F.float32, name="old_prob_%i" % i) for
                          i in range(len(self.onehot_probs))]
        if self.skip_weight is not None:
            self.loss += self.skip_weight * F.reduce_mean(self.onehot_skip_penaltys)
        if self.use_ppo_loss:
            self.loss += proximal_policy_optimization_loss(
                curr_prediction=self.onehot_probs,
                curr_onehot=self.input_arc_onehot,
                old_prediction=self.old_probs,
                old_onehotpred=self.input_arc_onehot,
                rewards=self.reward,
                advantage=self.advantage,
                clip_val=0.2)
        else:
            # onehot_log_prob: 1xbatch_size; advantage: batch_sizex1
            self.loss += F.reshape(F.tensordot(self.onehot_log_prob, self.advantage, axes=1), [])

        self.kl_div, self.ent = get_kl_divergence_n_entropy(curr_prediction=self.onehot_probs,
                                                            old_prediction=self.old_probs,
                                                            curr_onehot=self.input_arc_onehot,
                                                            old_onehotpred=self.input_arc_onehot)
        self.train_step = F.Variable(
            0, shape=(), dtype=F.int32, trainable=False, name="train_step")
        tf_variables = [var
                        for var in F.trainable_variables(scope=self.name)]

        self.train_op, self.lr, self.optimizer = F.get_train_op(
            loss=self.loss,
            variables=tf_variables,
            optimizer=self.optim_algo
        )

    def sample(self, *args, **kwargs):
        probs, onehots = self.session.run([self.sample_probs, self.sample_arc])
        return onehots, probs

    def train(self):
        self.buffer.finish_path()
        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):
            t = 0
            kl_sum = 0
            ent_sum = 0
            # get data from buffer
            for s_batch, p_batch, a_batch, ad_batch, nr_batch in self.buffer.get_data(self.batch_size):
                feed_dict = {self.input_arc[i]: a_batch[:, [i]]
                             for i in range(a_batch.shape[1])}
                feed_dict.update({self.advantage: ad_batch})
                feed_dict.update({self.old_probs[i]: p_batch[i]
                                  for i in range(len(self.old_probs))})
                feed_dict.update({self.reward: nr_batch})

                self.session.run(self.train_op, feed_dict=feed_dict)
                curr_loss, curr_kl, curr_ent = self.session.run([self.loss, self.kl_div, self.ent], feed_dict=feed_dict)
                aloss += curr_loss
                kl_sum += curr_kl
                ent_sum += curr_ent
                t += 1
                g_t += 1

                if kl_sum / t > self.kl_threshold and epoch > 0 and self.verbose > 0:
                    print("Early stopping at step {} as KL(old || new) = ".format(g_t), kl_sum / t)
                    return aloss / g_t

            if epoch % max(1, (self.train_pi_iter // 5)) == 0 and self.verbose > 0:
                print("Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}".format(
                    epoch, aloss / g_t,
                    kl_sum / t,
                    ent_sum / t)
                )
        return aloss / g_t

    def save_weights(self, filepath, **kwargs):
        weights = self.get_weights()
        with h5py.File(filepath, "w") as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name=self.params[i].name, data=d)

    def load_weights(self, filepath, **kwargs):
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(len(self.params)):
                key = self.params[i].name
                weights.append(hf.get(key).value)
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        weights = self.session.run(self.params)
        return weights

    def set_weights(self, weights, **kwargs):
        assign_ops = []
        for i in range(len(self.params)):
            assign_ops.append(F.assign(self.params[i], weights[i]))
        self.session.run(assign_ops)
