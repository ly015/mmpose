# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import PIPELINES2
from .base import BaseTransform


@PIPELINES2.register_module()
class HorizontalFlip(BaseTransform):

    def __init__(self, p) -> None:
        super().__init__()
        self.p = p

    def transform(self, results):
        raise NotImplementedError
