# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod


def with_cache(func):
    pass


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

        return {}
