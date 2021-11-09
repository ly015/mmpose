# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

# from mmpose.datasets_v2 import PIPELINES2, ApplyToSequence, HorizontalFlip
from mmpose.datasets_v2.pipelines.base import BaseTransform
from mmpose.datasets_v2.pipelines.utils import share_random_parameter


def test_share_random_parameter():

    class DummyTransform(BaseTransform):

        def get_random_parameter(self):
            return np.random.rand()

        def transform(self, results):
            results['random_param'] = self.get_random_parameter()
            return results

    transform = DummyTransform()

    with share_random_parameter(transform):
        results_1 = transform({})
        results_2 = transform({})
        np.testing.assert_equal(results_1['random_param'],
                                results_2['random_param'])

    results_1 = transform({})
    results_2 = transform({})
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(results_1['random_param'],
                                results_2['random_param'])
