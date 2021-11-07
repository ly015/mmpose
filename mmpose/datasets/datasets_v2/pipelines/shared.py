# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Optional, Union

from ..builder import PIPELINES_V2
from .base import Transform


@PIPELINES_V2.register_module()
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
                transform = PIPELINES_V2.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

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


@PIPELINES_V2.register_module()
class ApplyToList(Transform):

    def __init__(self,
                 transforms: list[Union[dict, Transform]],
                 input_mapping: Optional[dict] = None,
                 output_mapping: Optional[dict] = None,
                 inplace=False,
                 strict=False):
        super().__init__(input_mapping, output_mapping, inplace, strict)
        self.transforms = Compose(transforms)

    def get_default_input_mapping(self) -> dict:
        return {}

    def transform(self, **kwargs):
        raise NotImplementedError

    def split_input(self, data):
        # infer split number from input
        num_splits = 0
        key_rep = None
        for key in self._input_mapping.keys():
            if not self.strict and key not in data:
                continue

            assert isinstance(data[key], Sequence)
            if num_splits:
                if len(data[key]) != num_splits:
                    raise ValueError('Inconsistent length of inputs: '
                                     f'{num_splits} ({key_rep}) vs. '
                                     f'{len(data[key])} ({key})')
            else:
                num_splits = len(data[key])
                key_rep = key

        if not num_splits:
            raise RuntimeError('Fail to infer the input length.')

        splits = []
        for i in range(num_splits):
            split = data.copy()
            for key in self._input_mapping.keys():
                split[key] = data[key][i]
                splits.append(split)
        return splits

    def __call__(self, results: dict):

        input = self.collect_input(results, self._input_mapping)
        input_splits = self.split_input(input)

        output_splits = [self.transforms(split) for split in input_splits]

        # list of dict to dict of list
        output = {
            key: [split[key] for split in output_splits]
            for key in output_splits[0].keys()
        }

        if self._output_mapping:
            output = self.collect_output(output, self._output_mapping)

        results.update(output)
        return results
