# Copyright (c) OpenMMLab. All rights reserved.

import functools
import inspect
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Union

from .base import BaseTransform


class cacheable_method:

    def __init__(self, func):

        # Check `func` is to be bound as an instance method
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != 'self':
            raise TypeError(
                '@cacheable_method should only be used to decorate '
                'instance methods (the first argument is `self`).')

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, '_cacheable_methods'):
            setattr(owner, '_cacheable_methods', [])
        owner._cacheable_methods.append(self.__name__)

    def __call__(self, *args, **kwargs):
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag `self._cache_enabled`, which should be
        # set by the contextmanager `cache_random_parameters`
        cache_enabled = getattr(instance, '_cache_enabled', False)

        if cache_enabled:
            # Check `self._cache` which should be set by
            # the contextmanager `cache_random_parameters`
            assert hasattr(instance, '_cache')
            assert isinstance(instance._cache, dict)

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            return instance._cache[name]
        # Return function output
        return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        if self.instance_ref is None:
            self.instance_ref = weakref.ref(obj)
        return self


@contextmanager
def cache_random_params(transforms: Union[BaseTransform, Iterable]):

    def _cache_start(t: BaseTransform):
        setattr(t, '_cache_enabled', True)
        setattr(t, '_cache', {})

    def _cache_end(t: BaseTransform):
        delattr(t, '_cache_enabled')
        delattr(t, '_cache')

    def _apply(t: Union[BaseTransform, Iterable],
               func: Callable[[BaseTransform], None]):
        if isinstance(t, BaseTransform):
            if hasattr(t, '_cacheable_methods'):
                func(t)
        else:
            for _t in t:
                _apply(_t, func)

    try:
        _apply(transforms, _cache_start)
        yield
    finally:
        _apply(transforms, _cache_end)
