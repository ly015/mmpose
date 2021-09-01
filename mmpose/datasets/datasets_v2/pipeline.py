# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union


class Transform(metaclass=ABCMeta):

    def __init__(self, key_mapping: Optional[Union[dict, list]] = None):

        default_key_mapping = self._default_key_mapping
        if key_mapping is None:
            self._key_mappings = [copy.deepcopy(default_key_mapping)]
        else:
            self._key_mappings = key_mapping if isinstance(
                key_mapping, list) else [key_mapping]

        self._static_check_key_mappings(self._key_mappings)

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

    def _static_check_key_mappings(self, key_mappings):
        output_keys = set()
        use_default_output = False

        for i, key_mapping in enumerate(key_mappings):

            input_key_mapping = key_mapping.get('input', None)
            output_key_mapping = key_mapping.get('output', None)

            # check input_key_mapping
            if not input_key_mapping:
                raise KeyError(f'key_mappings[{i}] failed static check: '
                    'missing "input" field.')
                       
            # check unexpected input keys that doesn't match self.transform
            unexpected_keys = input_key_mapping.keys() - self.default_key_mapping['input'].keys()
            if unexpected_keys:
                raise ValueError(f'key_mappings[{i}] failed static check: '
                    f'Got unexpected keys {unexpected_keys}. '
                    f'Expect {self.default_key_mapping["input"].keys()}.')
            
            # check output
            if output_key_mapping is None:
                # check at most one output_key_mapping uses default
                # to valid output collision
                if use_default_output:
                    raise ValueError(f'key_mappings[{i}] failed static check: '
                        'Multiple key mappings used default output mapping, '
                        'while at most one is allowed.')
                use_default_output = True
            else:
                # check output key collision
                for outer_key in output_key_mapping.values():
                    if outer_key in output_keys:
                        raise ValueError(f'key_mappings[{i}] failed static check: '
                            f'"{outer_key}" has been used in another output mapping.')
                    output_keys.add(outer_key)

    def _dynamic_check_input_key_mapping(self, input_key_mapping, result):
        enexpected_keys = input_key_mapping.values() - result.keys()
        if enexpected_keys:
            raise 

    def _dynamic_check_output_key_mapping(self, output_key_mapping, output):
        pass

    def __call__(self, results:dict):
        for key_mapping  in self._key_mappings:
            kwargs = {inner_key:results[outer_key] for inner_key, outer_key in key_mapping['input'].items()}
            
            output = self.transform(**kwargs)

            if key_mapping['output']:
                output = {outer_key:output[inner_key] for inner_key, outer_key in key_mapping['output'].items()}
            
            results.update(output)
        return results

    @abstractmethod
    def transform(self, **kwargs) -> dict[str, Any]:
        pass
