# Copyright (c) OpenMMLab. All rights reserved.
import functools
from abc import ABCMeta, abstractclassmethod

import numpy as np


def with_cache(func):

    @functools.wraps(func)
    def wrapped_func(evaluator, *args, **kwargs):

        if evaluator.use_cache:

            key = func.__name__
            if key in evaluator.cache:
                return evaluator.cache[key].clone()
            else:
                value = func(evaluator, *args, **kwargs)
                evaluator.cache[key] = value.clone()
                return value

        else:
            return func(evaluator, *args, **kwargs)

    return wrapped_func


class BaseEvaluator(metaclass=ABCMeta):

    def __init__(self, use_cache=True):

        self.use_cache = use_cache
        self.cache = {}
        self.results = []

    def process(self, data: dict, output: dict):
        self.cache = {}

        result = self._process(data, output)
        self.results.append(result)

    @abstractclassmethod
    def _process(self, data: dict, output: dict) -> dict:
        """process one data batch and corresponding model output."""

    def evaluate(self) -> dict:

        results = {}

        assert self.results
        for k in self.results[0].keys():
            results[k] = np.mean([result[k] for result in self.results])

        self.results = []
        return results


class ComposedEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs):

        # build evaluators
        self._evaluators = []
        ...

        # share cache
        for evaluator in self._evaluators:
            evaluator.cache = self.cache

    def _process(self, data: dict, output: dict) -> dict:
        result = {}

        for evaluator in self._evaluators:
            _result = evaluator._process(data, output)
            result.update(_result)

        return result
