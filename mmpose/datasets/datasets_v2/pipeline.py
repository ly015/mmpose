# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union


class Transform(metaclass=ABCMeta):

    def __init__(self, key_mapping: Optional[Union[dict, list]] = None):

        default_key_mapping = self._default_key_mapping
        if key_mapping is None:
            self._key_mappings = [default_key_mapping]
        else:
            self._key_mappings = key_mapping if isinstance(
                key_mapping, list) else [key_mapping]

        self._check_key_mappings_static(self._key_mappings)

    @property
    def default_key_mapping(self):
        if not hasattr(self, '_default_key_mapping'):

            self._default_key_mapping = {
                'input': {
                    key: key
                    for key in inspect.signature(self.transform).parameters
                },
                'output': None
            }

        return copy.deepcopy(self._default_key_mapping)

    def _check_key_mappings_static(self, key_mappings):
        pass

    def _check_key_mappings_dynamic(self, key_mappings, results):
        pass

    def __call__(self, results):
        pass

    @abstractmethod
    def transform(self, **kwargs) -> dict[str, Any]:
        pass
