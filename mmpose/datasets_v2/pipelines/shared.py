# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

from ..builder import PIPELINES2
from .utils import cache_random_parameters


@PIPELINES2.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = PIPELINES2.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __iter__(self):
        """Allow easy iteration over the transform sequence."""
        return iter(self.transforms)

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES2.register_module()
class Remap():
    """A transform wrapper to remap and reorganize the input/output of the
    wrapped transforms (or sub-pipeline).

    Args:
        transforms (list[dict|callable]):
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

    def __init__(self,
                 transforms: list[Union[dict, Callable[[dict], dict]]],
                 input_mapping: Optional[dict] = None,
                 output_mapping: Optional[dict] = None,
                 inplace=False):

        self.inplace = inplace
        self.input_mapping = input_mapping

        if inplace:
            if output_mapping is not None:
                raise RuntimeError('Remap: the output_mapping must be None '
                                   'if `inplace` is set True')
            self.output_mapping = input_mapping
        else:
            self.output_mapping = output_mapping

        self.transforms = Compose(transforms)

    def remap_input(self, data: dict, input_mapping: dict) -> dict[str, Any]:
        """Remap inputs for the wrapped transforms by gathering and renaming
        data items according to the input_mapping.

        Args:
            data (dict): The original data dictionary of the pipeline
            input_mapping(dict):
        """

        def _remap(data, m):
            if isinstance(m, dict):
                # m is a dict {inner_key:outer_key, ...}
                return {k_in: _remap(data, k_out) for k_in, k_out in m.items()}
            if isinstance(m, (tuple, list)):
                # m is a list [outer_key1, outer_key2, ...]
                return m.__class__(_remap(data, e) for e in m)

            # m is an outer_key
            try:
                return data[m]
            except Exception as e:
                raise type(e)(f'Fail to collect {m} from data: {e}')

        collected = _remap(data, input_mapping)

        # Retain unmapped items
        inputs = data.copy()
        inputs.update(collected)

        return inputs

    def remap_output(self, data: dict, output_mapping: dict) -> dict[str, Any]:
        """Remap outputs from the wrapped transforms by gathering and renaming
        data items according to the output_mapping."""

        def _remap(data, m):
            if isinstance(m, dict):
                assert isinstance(data, dict)
                results = {}
                for k_in, k_out in m.items():
                    assert k_in in data
                    results.update(_remap(data[k_in], k_out))
                return results
            if isinstance(m, (list, tuple)):
                assert isinstance(data, (list, tuple))
                assert len(data) == len(m)
                return dict(zip(m, data))
            return dict(m=data)

        # Note that unmapped items are not retained, which is different from
        # the behavior in remap_input. This is to avoid original data items
        # being overwritten by intermediate namesakes
        return _remap(data, output_mapping)

    def __call__(self, results: dict) -> dict:

        inputs = self.remap_input(results, self.input_mapping)
        outputs = self.transforms(inputs)

        if self.output_mapping:
            outputs = self.remap_output(outputs, self.output_mapping)

        results.update(outputs)
        return results


@PIPELINES2.register_module()
class ApplyToMultiple(Remap):

    def __init__(self,
                 transforms: list[Union[dict, Callable[[dict], dict]]],
                 input_mapping: Optional[dict] = None,
                 output_mapping: Optional[dict] = None,
                 inplace: bool = False,
                 share_random_param: bool = False):
        super().__init__(
            transforms,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            inplace=inplace)

        self.share_random_param = share_random_param

    def scatter_sequence(self, data: dict) -> list[dict]:
        # infer split number from input
        seq_len = 0
        key_rep = None
        for key in self._input_mapping.keys():
            if not self.strict and key not in data:
                continue

            assert isinstance(data[key], Sequence)
            if seq_len:
                if len(data[key]) != seq_len:
                    raise ValueError('Got inconsistent sequence length: '
                                     f'{seq_len} ({key_rep}) vs. '
                                     f'{len(data[key])} ({key})')
            else:
                seq_len = len(data[key])
                key_rep = key

        if not seq_len:
            raise RuntimeError(
                'Fail to infer the sequence length. Please ensure that '
                'the input items are sequences with the same length.')

        scatters = []
        for i in range(seq_len):
            scatter = data.copy()
            for key in self._input_mapping.keys():
                scatter[key] = data[key][i]
            scatters.append(scatter)
        return scatters

    def __call__(self, results: dict):
        # Apply input remapping
        inputs = self.remap_input(results, self._input_mapping)

        # Scatter sequential inputs into a list
        inputs = self.scatter_sequence(inputs)

        # Control random parameter sharing with a contextmanager
        if self.share_random_param:
            cm = cache_random_parameters
        else:
            cm = nullcontext

        with cm(self.transforms):
            outputs = [self.transforms(**_input) for _input in inputs]

        # Collate output scatters (list of dict to dict of list)
        outputs = {
            key: [_output[key] for _output in outputs]
            for key in outputs[0].keys()
        }

        # Apply output remapping
        if self._output_mapping:
            outputs = self.remap_output(outputs, self._output_mapping)

        results.update(outputs)
        return results
