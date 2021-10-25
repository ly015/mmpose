# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional


class Transform(metaclass=ABCMeta):

    def __init__(self,
                 input_mapping: Optional[dict] = None,
                 output_mapping: Optional[dict] = None,
                 inplace=False,
                 strict=False):
        """
        Args:
            input_mapping (dict): A dict that defines the input key mapping.
                The keys corresponds to the inner key (i.e. kwargs of the
                `transform` method), and the values corresponds to the outer
                keys (i.e. the keys of the data/results).
            output_mapping(dict): A dict that defines the output key mapping.
                The keys corresponds to the inner key (i.e. the keys of the
                output dict of the `transform` method), and the values
                corresponds to the outer keys (i.e. the keys of the
                data/results).
        """

        self._inplace = inplace
        self._strict = strict

        self._input_mapping = copy.deepcopy(self.get_default_input_mapping())
        if input_mapping is not None:
            self._input_mapping.update(input_mapping)
            self._static_check_input_mapping(self._input_mapping)

        if output_mapping is None:
            if self.inplace:
                self._output_mapping = self._input_mapping
            else:
                self._output_mapping = None
        else:
            self._output_mapping = output_mapping

    @property
    def strict(self):
        return self._strict

    @property
    def inplace(self):
        return self._inplace

    def get_default_input_mapping(self) -> dict:
        if not hasattr(self, '_default_input_mapping'):

            self._default_input_mapping = {
                key: key
                for key in inspect.signature(self.transform).parameters
            }

        return self._default_input_mapping

    @property
    def get_default_output_mapping(self):
        if self.inplace:
            # inplace mode: output mapping is the same as input mapping
            return self.get_default_input_mapping()
        # no output mapping
        return None

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

    def collect_input(self, data: dict, input_mapping: dict) -> dict[str, Any]:
        """Collect input of `transform` function from the data according to the
        input mapping."""

        def _collect(data, m):
            if isinstance(m, dict):
                # m is a dict {inner_key:outer_key, ...}
                return {
                    k_in: _collect(data, k_out)
                    for k_in, k_out in m.items()
                }
            if isinstance(m, (tuple, list)):
                # m is a list [outer_key1, outer_key2, ...]
                return m.__class__(_collect(data, e) for e in m)

            # m is an outer_key
            try:
                return data[m]
            except Exception as e:
                raise type(e)(f'Fail to collect {m} from data: {e}')

        # if non-strict, skip the items missing in data and use default
        # argument value of `transform`.
        if not self.strict_key_mapping:
            input_mapping = {
                k: v
                for k, v in input_mapping.items() if v in data
            }

        return _collect(data, input_mapping)

    def collect_output(self, output: dict,
                       output_mapping: dict) -> dict[str, Any]:
        """Collect items from the `transform` output to update to the data
        flow."""

        def _collect(output, m):
            if isinstance(m, dict):
                assert isinstance(output, dict)
                results = {}
                for k_in, k_out in m.items():
                    assert k_in in output
                    results.update(_collect(output[k_in], k_out))
                return results
            if isinstance(m, (list, tuple)):
                assert isinstance(output, (list, tuple))
                assert len(output) == len(m)
                return dict(zip(m, output))
            return dict(m=output)

        return _collect(output, output_mapping)

    def __call__(self, results: dict):

        kwargs = self.collect_input(results, self._input_mapping)
        output = self.transform(**kwargs)

        if self._output_mapping:
            output = self.collect_output(output, self._output_mapping)

        results.update(output)
        return results

    @abstractmethod
    def transform(self, **kwargs) -> dict[str, Any]:
        pass


class RepeatableTransform(Transform):

    def __init__(self,
                 input_mapping: Optional[dict] = None,
                 output_mapping: Optional[dict] = None,
                 inplace=False,
                 strict=False,
                 broadcast=True):
        super().__init__(input_mapping, output_mapping, inplace, strict)
        self.broadcast = broadcast
        self._num_repeat = self.get_repeat_num()

    @property
    def broadcast(self):
        return self._broadcast

    def get_repeat_num(self) -> int:
        num_repeat = -1
        for k, v in self._input_mapping.items():
            if isinstance(v, (list, tuple)):
                if num_repeat == -1:
                    num_repeat = len(v)
                else:
                    raise ValueError(
                        'Invalid input_mapping for RepeatableTransform: '
                        'The length of repeated keys must be consistent, '
                        f'but got diverse lengths {num_repeat} vs {len(v)}')
            else:
                if not self.broadcast:
                    raise ValueError(
                        'Invalid input_mapping for RepeatableTransform: '
                        'only takes sequential outer keys for '
                        'input key mapping when broadcast==False, '
                        f'got single outer key "{k}"')

        if num_repeat > 0:
            if not self._output_mapping:
                raise ValueError(
                    'Invalid output_mapping for RepeatableTransform: '
                    'output_mapping cannot be "None" when input_mapping '
                    'is repeatable.')
            for k, v in self._output_mapping.items():
                if len(v) != num_repeat:
                    raise ValueError(
                        'Invalid output_mapping for RepeatableTransform: '
                        'The length of repeated keys must be consistent with '
                        f' input_mapping but got diverse lengths {num_repeat} '
                        f' vs {len(v)}')

        return num_repeat

    def split_key_mapping(self, key_mapping):
        return {}

    def __call__(self, results: dict):

        if self._num_repeat == -1:
            return super().__call__(results)
        else:
            sub_input_mappings = self.split_key_mapping(self._input_mapping)
            sub_output_mappings = self.split_key_mapping(self._output_mapping)

            output = {}
            for _input_mapping, _output_mapping in zip(sub_input_mappings,
                                                       sub_output_mappings):
                kwargs = self.collect_input(results, _input_mapping)
                _output = self.transform(**kwargs)

                _output = self.collect_output(_output, _output_mapping)

            output.update(_output)

        results.update(output)
