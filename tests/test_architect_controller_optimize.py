"""Test architect optimizer"""

import copy
import logging
import sys
import tempfile
import unittest

import numpy as np
from parameterized import parameterized_class

from amber import architect
from amber import backend as F
# need to test for seamlessly connecting to manager's architectureDecoder as well. FZZ 2022.5.4
from amber.modeler import architectureDecoder
from amber.utils import testing_utils

logging.disable(sys.maxsize)


class TestModelSpace(testing_utils.TestCase):
    def test_conv1d_model_space(self):
        model_space = architect.ModelSpace()
        num_layers = 2
        out_filters = 8
        layer_ops = [
                architect.Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
                architect.Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
                architect.Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
                architect.Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
        ]
        # init
        for i in range(num_layers):
            model_space.add_layer(i, copy.copy(layer_ops))
        self.assertLen(model_space, num_layers)
        self.assertLen(model_space[0], 4)
        # Add layer
        model_space.add_layer(2, copy.copy(layer_ops))
        self.assertLen(model_space, num_layers + 1)
        # Add op
        model_space.add_state(2, architect.Operation('identity', filters=out_filters))
        self.assertLen(model_space[2], 5)
        # Delete op
        model_space.delete_state(2, 4)
        self.assertLen(model_space[2], 4)
        # Delete layer
        model_space.delete_layer(2)
        self.assertLen(model_space, num_layers)


class TestGeneralController(testing_utils.TestCase):
    def setUp(self):
        super(TestGeneralController, self).setUp()
        self.session = F.Session()
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=12, num_pool=3)
        self.controller = architect.controller.RecurrentRLController(
            model_space=self.model_space,
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            session=self.session,
            train_pi_iter=2,
            lstm_size=64,
            lstm_num_layers=1,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=None,
        )
        self.tempdir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        super(TestGeneralController, self).tearDown()
        self.tempdir.cleanup()

    def test_get_architecture(self):
        act, prob = self.controller.sample()
        self.assertIsInstance(act, np.ndarray)
        self.assertIsInstance(prob, list)
        # initial probas should be close to uniform
        i = 0
        for layer_id in range(len(self.model_space)):
            # operation
            pr = prob[i]
            self.assertAllClose(pr.flatten(), [1./len(pr.flatten())] * len(pr.flatten()), atol=0.05)
            # skip connection
            if layer_id > 0:
                pr = prob[i + 1]
                self.assertAllClose(pr.flatten(), [0.5] * len(pr.flatten()), atol=0.05)
                i += 1
            i += 1

    @unittest.skipIf(F.mod_name=='tensorflow_1', "only implemented in dynamic backend")
    def test_optimizer_dynamic(self):
        a1, p1 = self.controller.sample()
        a2, p2 = self.controller.sample()
        a_batch = np.array([a1, a2]).T
        p_batch = [np.concatenate(x) for x in zip(*[p1, p2])]
        self.controller.evaluate(input_arc=a_batch)
        old_log_probs, old_probs, _, _ = F.to_numpy(self.controller.evaluate(input_arc=a_batch))
        losses = []
        max_iter = 100
        for i in range(max_iter):
            loss, _, _ = self.controller.train_step(
                input_arc=a_batch, 
                advantage=F.Variable([1,-1], trainable=False),
                old_probs=p_batch
            )
            if i % (max_iter//5) == 0:
                losses.append(F.to_numpy(loss))
        new_log_probs, new_probs, _, _ = F.to_numpy(self.controller.evaluate(input_arc=a_batch))
        # loss should decrease over time
        self.assertLess(losses[-1], losses[0])
        # 1st index positive reward should decrease/minimize its loss
        self.assertLess(new_log_probs[0], old_log_probs[0])
        # 2nd index negative reward should increase/increase the loss
        self.assertLess(old_log_probs[1], new_log_probs[1])
    
    @unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in static/TF1 backend")
    def test_optimize_static(self):
        a1, p1 = self.controller.sample()
        a2, p2 = self.controller.sample()
        a_batch = np.array([a1, a2])
        p_batch = [np.concatenate(x) for x in zip(*[p1, p2])]
        feed_dict = {self.controller.input_arc[i]: a_batch[:, [i]]
                     for i in range(a_batch.shape[1])}
        feed_dict.update({self.controller.old_probs[i]: p_batch[i]
                    for i in range(len(self.controller.old_probs))})             
        # add a pseudo reward - the first arc is 1. ; second arc is -1.
        feed_dict.update({self.controller.advantage: np.array([1., -1.]).reshape((2, 1))})
        feed_dict.update({self.controller.reward: np.array([1., 1.]).reshape((2, 1))})
        old_loss = self.session.run(self.controller.onehot_log_prob, feed_dict)
        losses = []
        max_iter = 50
        for i in range(max_iter):
            self.session.run(self.controller.train_op, feed_dict=feed_dict)
            if i % (max_iter//5) == 0:
                losses.append(self.session.run(self.controller.loss, feed_dict))
        new_loss = self.session.run(self.controller.onehot_log_prob, feed_dict)
        # loss should decrease over time
        self.assertLess(losses[-1], losses[0])
        # 1st index positive reward should decrease/minimize its loss
        self.assertLess(new_loss[0], old_loss[0])
        # 2nd index negative reward should increase/increase the loss
        self.assertLess(old_loss[1], new_loss[1])

    def test_train(self):
        # random store some entries
        arcs = []
        probs = []
        rewards = []
        for _ in range(10):
            arc, prob = self.controller.sample()
            arcs.append(arc)
            probs.append(prob)
            rewards.append(np.random.random(1)[0])
        for arc, prob, reward in zip(*[arcs, probs, rewards]):
            self.controller.store(prob=prob, action=arc,
                                    reward=reward)
        # train
        old_loss = self.controller.train()
        for arc, prob, reward in zip(*[arcs, probs, rewards]):
            self.controller.store(prob=prob, action=arc,
                                    reward=reward)
        new_loss = self.controller.train()
        self.assertLess(new_loss, old_loss)



if __name__ == '__main__':
    unittest.main()

