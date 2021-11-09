# Copyright (c) OpenMMLab. All rights reserved.

from .builder import PIPELINES2
from .datasets import *  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403

__all__ = datasets.__all__ + pipelines.__all__ + ['PIPELINES2']  # noqa: F405
