# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from ._typing import CONFIG
from .base_dataset import BaseDataset


class TopDownCOCODataset(BaseDataset):

    def __init__(self,
                 ann_file: str,
                 pipeline: list[CONFIG],
                 bbox_file: Optional[str] = None,
                 use_gt_bbox: bool = True,
                 data_root: Optional[str] = None,
                 meta_cfg: Optional[CONFIG] = None,
                 test_mode: bool = False):

        self.bbox_file = bbox_file
        self.use_gt_bbox = use_gt_bbox

        super().__init__(
            ann_file, pipeline, data_root, meta_cfg, test_mode=test_mode)

    def load_annotations(self) -> list[dict]:

        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bboxes and keypoints
            return self._load_coco_annotations(self.ann_file)
        else:
            # use bboxes from detection results
            return self._load_detection_results(self.bbox_file)
