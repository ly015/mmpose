# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar

from torch.utils.data import Dataset

from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose
from ._typing import CONFIG

T = TypeVar('T')  # type of data sample


class BaseDataset(Dataset[T], metaclass=ABCMeta):
    """Base dataset class for pose estimation."""

    def __init__(self,
                 ann_file: str,
                 pipeline: list[CONFIG],
                 data_root: Optional[str],
                 meta_cfg: Optional[CONFIG],
                 test_mode: bool = False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.meta_cfg = copy.deepcopy(meta_cfg)
        self.test_mode = test_mode

        # load meta information
        # TBD: dataset meta information should be from configs or coded in
        # the classes?
        self.meta_data = self.load_meta_data(self.meta_cfg)

        # load data information
        self.data_infos = self.load_annotations(self.ann_file)

        # build pipeline
        self.pipeline = Compose(pipeline)

    @abstractmethod
    def load_annotations(self) -> list[dict]:
        pass

    def load_meta_data(self, meta_cfg: Optional[CONFIG]):
        dataset_info = DatasetInfo(meta_cfg)

        meta_data = dict(
            num_joints=dataset_info.keypoint_num,
            flip_pairs=dataset_info.flip_pairs,
            flip_index=dataset_info.flip_index,
            upper_body_ids=dataset_info.upper_body_ids,
            lower_body_ids=dataset_info.lower_body_ids,
            joint_weights=dataset_info.joint_weights,
            skeleton=dataset_info.skeleton,
            sigmas=dataset_info.sigmas,
            dataset_name=dataset_info.dataset_name)

        return meta_data

    def _prepare_train_data(self, index) -> T:
        results = copy.deepcopy(self.data_infos[index])
        meta_data = copy.deepcopy(self.meta_data)
        results.update(meta_data)
        return self.pipeline(results)

    def _prepare_test_data(self, index) -> T:
        results = copy.deepcopy(self.data_infos[index])
        meta_data = copy.deepcopy(self.meta_data)
        results.update(meta_data)
        return self.pipeline(results)

    def __getitem__(self, index) -> T:
        if self.test_mode:
            return self._prepare_test_data(index)
        else:
            return self._prepare_train_data(index)

    def __len__(self) -> int:
        return len(self.data_infos)

    def evaluate(self, outputs, logger, **eval_kwargs):
        warnings.warn(
            'Dataset and evluation will be decoupled in OpenMMLab v2.0, '
            'and the `evaluate` method of dataset class will be deprecated '
            'in the future.', UserWarning)
