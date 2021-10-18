# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from .pipeline import Transform


class Resize(Transform):

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False,
                 key_mapping=None):

        super().__init__(key_mapping)

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        ...

    @staticmethod
    def random_sample(img_scales):
        ...

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        ...

    def _random_scale(self):
        ...
        # return scale, scale_idx

    def _resize_img(self, img, scale):
        """
        Returns:
            img
            meta (dict):
                - img_shape
                - pad_shape
                - scale_factor
                - keep_ratio
        """
        ...
        # return img, meta

    def _resize_bbox(self, bbox, scale_factor, img_shape):
        ...
        # return bbox

    def _resize_mask(self, mask, scale, img_shape):
        ...
        # return mask

    def _resize_seg(self, seg, scale):
        ...
        # return seg

    def __repr__(self):
        ...

    def transform(self, imgs, bboxes=None, masks=None, segs=None, scale=None):

        output = {}

        assert imgs

        if scale is None:
            scale = self._random_scale()

        # resize images
        imgs_resized = []
        meta = dict(scale=scale)
        for img in imgs:
            img_resized, meta = self._resize_img(img, scale)
            imgs_resized.append(img_resized)

        output['imgs'] = imgs_resized
        output.update(meta)

        img_shape = meta['img_shape']
        scale_factor = meta['scale_factor']

        # resize bboxes
        if bboxes:
            bboxes_resized = [
                self._resize_bbox(bbox, scale_factor, meta['img_shape'])
                for bbox in bboxes
            ]
            output['bboxes'] = bboxes_resized

        # resize masks
        if masks:
            masks_resized = [
                self._resize_mask(mask, scale, img_shape) for mask in masks
            ]
            output['masks'] = masks_resized

        # resize segs
        if segs:
            segs_resized = [self._resize_seg(seg, scale) for seg in segs]
            output['segs'] = segs_resized

        return output
