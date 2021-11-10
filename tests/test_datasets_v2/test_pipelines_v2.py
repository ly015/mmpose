# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import config  # noqa F401

from mmpose.datasets_v2 import PIPELINES2  # noqa F401
from mmpose.datasets_v2.pipelines.base import BaseTransform
from mmpose.datasets_v2.pipelines.utils import (AllowCache, allow_cache,
                                                cache_random_parameters,
                                                share_random_parameters)


def test_share_random_parameter():

    class DummyTransform(BaseTransform):

        def get_random_parameter(self):
            return np.random.rand()

        def transform(self, results):
            results['random_param'] = self.get_random_parameter()
            return results

    transform = DummyTransform()

    with share_random_parameters(transform):
        results_1 = transform({})
        results_2 = transform({})
        np.testing.assert_equal(results_1['random_param'],
                                results_2['random_param'])

    results_1 = transform({})
    results_2 = transform({})
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(results_1['random_param'],
                                results_2['random_param'])


def test_cache_random_parameters():

    class DummyTransform(BaseTransform):

        @AllowCache
        def get_random_a(self):
            return np.random.rand()

        @allow_cache
        def get_random_b(self):
            return np.random.rand()

        def transform(self, results):
            results['random_param'] = self.get_random_a() + self.get_random_b()
            return results

    transform = DummyTransform()

    assert hasattr(DummyTransform, '_cacheable_methods')
    assert 'get_random_a' in DummyTransform._cacheable_methods

    with cache_random_parameters(transform):
        results_1 = transform({})
        results_2 = transform({})
        np.testing.assert_equal(results_1['random_param'],
                                results_2['random_param'])

    results_1 = transform({})
    results_2 = transform({})
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(results_1['random_param'],
                                results_2['random_param'])
