# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any


class BaseTransform(metaclass=ABCMeta):

    def __call__(self, results: dict) -> dict:

        return self.transform(results)

    def get_random_parameter(self, results) -> Any:
        pass

    @abstractmethod
    def transform(self, results):
        pass
