# Copyright (c) OpenMMLab. All rights reserved.

import inspect
import weakref
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Union

import mmcv


class BaseTransform(metaclass=ABCMeta):

    def __call__(self, results: dict) -> dict:

        return self.transform(results)

    def get_random_parameter(self, results):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, results):
        pass


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
def share_random_parameters(transforms: Union[BaseTransform, Iterable]):

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


def allow_cache(func):

    # Check `func` is to be bound as an instance method
    func_args = inspect.getfullargspec(func).args
    if len(func_args) == 0 or func_args[0] != 'self':
        raise TypeError('@allow_cache should only be used to decorate '
                        'instance methods (the first argument is `self`).')

    # Register to the class' random parameter generators
    name = func.__name__

    @wraps(func)
    def wrapped(self, *args, **kwargs):

        # Check the flag `self._cache_enabled`, which should be
        # set by the contextmanager `cache_random_parameters`
        cache_enabled = getattr(self, '_cache_enabled', False)

        if cache_enabled:
            # Check `self._cache` which should be set by
            # the contextmanager `cache_random_parameters`
            assert hasattr(self, '_cache')
            assert isinstance(self._cache, dict)

            if name not in self._cache:
                self._cache[name] = func(self, *args, **kwargs)

            # Return the cached value
            return self._cache[name]
        # Return function output
        return func(self, *args, **kwargs)

    return wrapped
