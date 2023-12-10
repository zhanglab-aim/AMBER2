"""
pmbga = Probabilistic Model-Building Genetic Algorithms

From Wiki:
Estimation of distribution algorithms (EDAs), sometimes called probabilistic model-building genetic algorithms (
PMBGAs),[1] are stochastic optimization methods that guide the search for the optimum by building and sampling
explicit probabilistic models of promising candidate solutions. Optimization is viewed as a series of incremental
updates of a probabilistic model, starting with the model encoding an uninformative prior over admissible solutions
and ending with the model that generates only the global optima.
"""

import sys
import os
import numpy as np
from collections import defaultdict
from .bayes_prob import *
from ..model_space import ModelSpace, Operation


class PopulationBuffer:
    """Population buffer for working with genetic algorithms"""

    def __init__(
        self,
        max_size,
        discount_factor=0.0,
        ewa_beta=None,
        is_squeeze_dim=False,
        rescale_advantage_by_reward=False,
        clip_advantage=10.0,
    ):
        self.max_size = max_size
        self.ewa_beta = (
            ewa_beta if ewa_beta is not None else float(1 - 1.0 / self.max_size)
        )
        self.discount_factor = discount_factor
        self.is_squeeze_dim = is_squeeze_dim
        self.rescale_advantage_by_reward = rescale_advantage_by_reward
        self.clip_advantage = clip_advantage
        self.r_bias = None

        # short term buffer storing single trajectory
        self.state_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.reward_buffer = []

        # long_term buffer
        self.lt_sbuffer = []  # state
        self.lt_abuffer = []  # action
        self.lt_pbuffer = []  # prob
        self.lt_adbuffer = []  # advantage
        self.lt_nrbuffer = []  # lt buffer for non discounted reward
        self.lt_rmbuffer = []  # reward mean buffer

    def store(self, state=None, prob=None, action=None, reward=None, *args, **kwargs):
        if state is not None:
            self.state_buffer.append(state)
        if prob is not None:
            self.prob_buffer.append(prob)
        if action is not None:
            self.action_buffer.append(action)
        if reward is not None:
            self.reward_buffer.append(reward)

    def finish_path(self, state_space, global_ep, working_dir, *args, **kwargs):
        # self.ewa_beta = 0
        # sort acts by reward
        act_reward_pairs = [
            (a, r) for a, r in zip(*[self.action_buffer, self.reward_buffer])
        ]
        act_reward_pairs = sorted(act_reward_pairs, key=lambda x: x[1], reverse=True)
        # append to long-term
        if self.lt_abuffer is None:
            self.lt_abuffer = [[x[0] for x in act_reward_pairs]]
            self.lt_nrbuffer = [[x[1] for x in act_reward_pairs]]
        else:
            self.lt_abuffer.append([x[0] for x in act_reward_pairs])
            self.lt_nrbuffer.append([x[1] for x in act_reward_pairs])
        # update r-bias
        if self.r_bias is None:
            self.r_bias = np.mean(self.lt_nrbuffer[-1])
        else:
            # print(f"r_bias = {self.r_bias} * {self.ewa_beta} + {(1. - self.ewa_beta)} * {np.mean(self.lt_nrbuffer[-1])}")
            self.r_bias = self.r_bias * self.ewa_beta + (1.0 - self.ewa_beta) * np.mean(
                self.lt_nrbuffer[-1]
            )
        # write out
        with open(os.path.join(working_dir, "buffers.txt"), mode="a+") as f:
            action_readable_str = ",".join([str(x) for x in self.action_buffer[-1]])
            f.write(
                "-" * 80
                + "\n"
                + "Episode:%d\tReward:%.4f\tR_bias:%.4f\tAdvantage:NA\n"
                % (
                    global_ep,
                    self.reward_buffer[-1],
                    self.r_bias,
                )
                + "\tAction:%s\n\tProb:NA\n" % (action_readable_str,)
            )
        # remove tailing buffer to max_size
        if len(self.lt_abuffer) > self.max_size:
            self.lt_abuffer = self.lt_abuffer[-self.max_size :]
            self.lt_nrbuffer = self.lt_nrbuffer[-self.max_size :]
        self.action_buffer, self.reward_buffer = [], []

    def get_data(self, bs, shuffle=True):
        """return any architectures whose reward is higher than the moving award reward `r_bias` in batches.

        Depending on the NAS application, this ad hoc survival selection criterion can be either good or bad.
        In building kinetic-interpretable neural networks, it works good; however, if one needs customization,
        subclassing `amber.architect.Buffer` should be considered.
        """
        lt_abuffer = np.concatenate(self.lt_abuffer, axis=0)
        lt_nrbuffer = np.concatenate(self.lt_nrbuffer, axis=0)
        lt_abuffer = lt_abuffer[lt_nrbuffer >= self.r_bias]
        lt_nrbuffer = lt_nrbuffer[lt_nrbuffer >= self.r_bias]
        if shuffle is True:
            slice_ = np.random.choice(
                lt_nrbuffer.shape[0], size=lt_nrbuffer.shape[0], replace=False
            )
            lt_abuffer = lt_abuffer[slice_]
            lt_nrbuffer = lt_nrbuffer[slice_]

        for i in range(0, len(lt_nrbuffer), bs):
            b = min(i + bs, len(lt_nrbuffer))
            p_batch = None
            a_batch = lt_abuffer[i:b]
            yield None, p_batch, a_batch, None, lt_nrbuffer[i:b]


class ProbaModelBuildGeneticAlgo(object):
    def __init__(self, model_space, buffer_size=1, batch_size=100, *args, **kwargs):
        """the workhorse for building model hyperparameters using Bayesian genetic algorithms

        I tried to match the arguments with BaseController, whenever possible; however, this means some arguments will
        need better naming and/or documentations in the future, as a unified nomenclature to make better sense

        Parameters
        ----------
        model_space : amber.architect.ModelSpace
            model space with computation operation's hyperparameters follow a Bayesian distribution, instead of fixed
            values
        buffer_type : str, or callable
            buffer identifier or callable to parse to amber.buffer.get_buffer
        buffer_size : int
            population size; how many history episodes/epochs to keep in memory
        batch_size : int
            within each epidose/epoch, how many individuals to keep
        """
        self.model_space = model_space
        assert isinstance(self.model_space, ModelSpace)
        for i in range(len(self.model_space)):
            assert len(self.model_space[i]) == 1, (
                "pmbga does not support more than 1 operations per layer; check %i" % i
            )
        ewa_beta = kwargs.get("ewa_beta", 0)
        if ewa_beta is None or ewa_beta == "auto":
            ewa_beta = 1 - 1.0 / buffer_size
        self.buffer = PopulationBuffer(
            max_size=buffer_size,
            ewa_beta=ewa_beta,
            discount_factor=0.0,
            is_squeeze_dim=True,
        )
        self.batch_size = batch_size
        self.model_space_probs = self._get_probs_in_model_space()

    def sample(self):
        arcs_tokens = []
        for i in range(len(self.model_space)):
            op = self.model_space[i][0]
            tmp = {}
            for k, v in op.Layer_attributes.items():
                if isinstance(v, BayesProb):
                    v_samp = v.sample()
                    tmp[k] = v_samp
                else:
                    tmp[k] = v
            arcs_tokens.append(Operation(op.Layer_type, **tmp))
        return arcs_tokens, None

    def store(self, action, reward):
        """
        Parameters
        ----------
        action : list
            A list of architecture tokens sampled from posterior distributions

        reward : float
            Reward for this architecture, as evaluated by ``amber.architect.manager``
        """
        self.buffer.store(state=[], prob=[], action=action, reward=reward)
        return self

    def _get_probs_in_model_space(self):
        d = {}
        for i in range(len(self.model_space)):
            for k, v in self.model_space[i][0].Layer_attributes.items():
                if isinstance(v, BayesProb):
                    d[(i, self.model_space[i][0].Layer_type, k)] = v
        return d

    def _parse_probs_in_arc(self, arc, update_dict):
        for i in range(len(self.model_space)):
            obs = arc[i]
            for k, v in self.model_space[i][0].Layer_attributes.items():
                if isinstance(v, BayesProb):
                    update_dict[v].append(obs.Layer_attributes[k])
        return update_dict

    def train(self, episode, working_dir):
        try:
            self.buffer.finish_path(
                state_space=self.model_space, global_ep=episode, working_dir=working_dir
            )
        except Exception as e:
            print("cannot finish path in buffer because: %s" % e)
            sys.exit(1)
        # parse buffers; only arcs w/ reward > r_bias will be yielded
        gen = self.buffer.get_data(bs=self.batch_size)
        arcs = []
        rewards = []
        for data in gen:
            arcs.extend(data[2])
            rewards.extend(data[4])
        # match sampled values to priors
        update_dict = defaultdict(list)
        for a, r in zip(*[arcs, rewards]):
            update_dict = self._parse_probs_in_arc(arc=a, update_dict=update_dict)

        # update posterior with data
        for k, v in update_dict.items():
            k.update(data=v)
        print(
            "datapoints: ",
            len(v),
            "/ total: ",
            len(np.concatenate(self.buffer.lt_nrbuffer, axis=0)),
        )
