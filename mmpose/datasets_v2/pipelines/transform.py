# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ..builder import PIPELINES2
from .base import BaseTransform


@PIPELINES2.register_module()
class HorizontalFlip(BaseTransform):

    def __init__(self, p) -> None:
        super().__init__()
        self.p = p

    def get_flip_flag(self):
        return np.random.rand() < self.p

    def transform(self, results):

        flip = self.get_flip_flag()
        if flip:
            results['img'] = results['img'][:, ::-1]
