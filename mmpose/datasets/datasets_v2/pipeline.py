# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional


class Transform(metaclass=ABCMeta):

    def __init__(self,
                 key_mapping: Optional[dict] = None,
                 strict_key_mapping=False):

        self.strict_key_mapping = strict_key_mapping

        if key_mapping is None:
            self._key_mapping = copy.deepcopy(self._default_key_mapping)
        else:
            self._key_mapping = key_mapping

        self._static_check_key_mapping(self._key_mapping)

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

        return self._default_key_mapping

    def _static_check_input_mapping(self, input_mapping):

        if input_mapping is None:
            raise KeyError(
                'Static input_mapping check failed: missing "input".')

        default_input_mapping = self.default_key_mapping['input']
        unexpected_keys = input_mapping.keys() - default_input_mapping.keys()

        if unexpected_keys:
            raise ValueError('Static input_mapping check failed: '
                             f'Got unexpected keys {unexpected_keys}, '
                             f'expect {default_input_mapping.keys()}.')

    def _dynamic_check_input_mapping(self, input_mapping, result):
        missing_keys = input_mapping.values() - result.keys()
        if missing_keys:
            raise RuntimeError('Dynamic input_mapping check failed: '
                               f'keys {missing_keys} missing in data.')

    def _dynamic_check_output_key_mapping(self, output_mapping, output):

        if output_mapping is None:
            return

        missing_keys = output_mapping.values() - output.keys()
        if missing_keys:
            raise RuntimeError('Dynamic output_mapping check failed.'
                               f'keys {missing_keys} missing in the '
                               'transform output')

    def collect_input(self, data: dict,
                      input_mapping: dict[str, Any]) -> dict[str, Any]:

        def _collect(data, input_mapping):
            if isinstance(input_mapping, dict):
                return {k: _collect(data, v) for k, v in input_mapping.items()}
            if isinstance(input_mapping, (tuple, list)):
                return input_mapping.__class__(
                    (_collect(data, e) for e in input_mapping))
            try:
                return data[input_mapping]
            except Exception as e:
                raise type(e)(
                    f'Fail to apply input_mapping: {input_mapping}: {e}')

        return _collect(data, input_mapping)

    def collect_output(self, output: dict,
                       output_mapping: dict[str, Any]) -> dict[str, Any]:
        ...

    def __call__(self, results: dict):

        key_mapping = self._key_mapping

        self._dynamic_check_input_mapping(key_mapping, results)

        kwargs = self.collect_input(results, key_mapping['input'])
        output = self.transform(**kwargs)

        if key_mapping.get('output', None):
            self._dynamic_check_output_key_mapping(key_mapping, output)
            # output = {
            #     outer_key: output[inner_key]
            #     for inner_key, outer_key in key_mapping['output'].items()
            # }
            output = self.collect_output(output, key_mapping['output'])
        results.update(output)
        return results

    @abstractmethod
    def transform(self, **kwargs) -> dict[str, Any]:
        pass
