# Copyright (c) OpenMMLab. All rights reserved.

import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Union

import mmcv

from .base import BaseTransform


def _set_method_output_sharing(method):
    name = method.__name__
    stash_name = f'_stashed_{name}'
    cache_name = f'_cache_{name}'
    instance_ref = weakref.ref(method.__self__)
    cls = instance_ref().__class__
    func = method.__func__

    # Check if output sharing has already been set for the method.
    if hasattr(instance_ref(), stash_name) or hasattr(instance_ref(),
                                                      cache_name):
        exist_name = stash_name if hasattr(instance_ref(),
                                           stash_name) else cache_name
        raise RuntimeError(
            f'Attempt to set output sharing for method {name} '
            f'of {cls} instance, but the attribute {exist_name} '
            'already exists. Please do not set it repetitively.')

    # Stash the original method
    setattr(instance_ref(), stash_name, method)

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        instance = instance_ref()
        if not hasattr(instance, cache_name):
            setattr(instance, cache_name, func(*args, **kwargs))
        return getattr(instance, cache_name)

    wrapped_method = wrapped_func.__get__(instance_ref(), cls)

    # Replace with the wrapped method
    setattr(instance_ref(), name, wrapped_method)


def _unset_method_output_sharing(method):
    name = method.__name__
    stash_name = f'_stashed_{name}'
    cache_name = f'_cache_{name}'
    instance_ref = weakref.ref(method.__self__)

    # Get the stashed original method
    assert hasattr(instance_ref(), stash_name)
    stashed = getattr(instance_ref(), stash_name)
    delattr(instance_ref(), stash_name)

    # Clear the output cache
    if hasattr(instance_ref(), cache_name):
        delattr(instance_ref(), cache_name)

    setattr(instance_ref(), name, stashed)


@contextmanager
def share_random_parameter(transforms: Union[BaseTransform, Iterable]):

    def _apply(t: Union[BaseTransform, Iterable], func: Callable[[Callable],
                                                                 None]):
        if isinstance(t, BaseTransform):
            method = t.get_random_parameter

            # Only apply to transforms that has overridden the
            # `get_random_parameter` method.
            if mmcv.is_method_overridden(method.__name__, BaseTransform,
                                         t.__class__):
                func(method)

        else:
            assert isinstance(t, Iterable)
            for _t in t:
                _apply(_t, func)

    # Set random parameter sharing
    _apply(transforms, _set_method_output_sharing)

    try:
        yield
    finally:
        # Unset random parameter sharing
        _apply(transforms, _unset_method_output_sharing)
