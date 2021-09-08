# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar

from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose

T = TypeVar('T')  # type of data sample
S = TypeVar('S')  # type of data_infos


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset class for pose estimation."""

    def __init__(self,
                 ann_file: str,
                 pipeline: list,
                 data_root: Optional[str] = None,
                 test_mode: bool = False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode

        # load data
        # data (list): all dataset samples
        # meta_data (dict): dataset meta information
        self.data_infos, self.meta_data = self.load_annotations(self.ann_file)

        # build pipeline
        self.pipeline = Compose(pipeline)

    @abstractmethod
    def load_annotations(self, ann_file: str) -> tuple[list[dict], dict]:
        pass

    @abstractmethod
    def _prepare_train_data(self, sample: S) -> dict:
        pass

    @abstractmethod
    def _prepare_test_data(self, sample: S) -> dict:
        pass

    def __getitem__(self, index) -> T:
        data_info = copy.deepcopy(self.data_infos[index])

        if self.test_mode:
            result = self._prepare_test_data(data_info)
        else:
            result = self._prepare_train_data(data_info)

        return self.pipeline(result)

    def __len__(self) -> int:
        return len(self.data_infos)
