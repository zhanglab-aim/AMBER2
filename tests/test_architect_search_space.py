import copy
import logging
import sys
import tempfile
import unittest

import numpy as np
from parameterized import parameterized_class

from amber import architect
from amber.utils import testing_utils

logging.disable(sys.maxsize)

# TODO: TestVariableSpace

class TestModelSpace(testing_utils.TestCase):
    def test_conv1d_model_space(self):
        model_space = architect.OperationSpace()
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