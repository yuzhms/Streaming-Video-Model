import copy
import numpy as np
from numpy import random
import cv2
import torch
import mmcv
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import Mosaic, RandomAffine, MixUp, YOLOXHSVRandomAug
from mmdet.datasets.pipelines.test_time_aug import MultiScaleFlipAug
from mmdet.datasets.pipelines.loading import FilterAnnotations
from mmdet.core import find_inside_bboxes
from mmdet.datasets.pipelines import to_tensor

from ..pipelines.formatting import SeqDefaultFormatBundle


@PIPELINES.register_module()
class SeqMosaic(Mosaic):
    """Sequence mosaic augmentation.

    This operation uses the same augmentation parameters for all images in the sequence
    """

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        if random.uniform(0, 1) > self.prob:
            return results
        center_position = None
        outs = []
        for frame_idx, _results in enumerate(results):
            _results, center_position = self._mosaic_transform(
                _results, center_position)
            outs.append(_results)
        return outs

    def _mosaic_transform(self, results, center_position=None):
        """Mosaic transform function for sequence data,
           which has the `center_position` parameter to share the same
           augmentation parameters for all images in the sequence.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        if center_position is None:
            center_x = int(
                random.uniform(*self.center_ratio_range) * self.img_scale[1])
            center_y = int(
                random.uniform(*self.center_ratio_range) * self.img_scale[0])
            center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results, center_position


@PIPELINES.register_module()
class SeqRandomAffine(RandomAffine):
    """Sequence random affine augmentation.

    This operation uses the same augmentation parameters for all images in the sequence
    """
    def __call__(self, results):
        _img = results[0]['img']
        height = _img.shape[0] + self.border[0] * 2
        width = _img.shape[1] + self.border[1] * 2
        warp_matrix, scaling_ratio = self._get_affine_param(height, width)

        outs = []
        for _results in results:
            _results = self._transform_img(_results, width, height, warp_matrix)
            _results = self._transform_bboxes(_results, width, height, warp_matrix, scaling_ratio)
            outs.append(_results)
        return outs

    def _transform_img(self, results, width, height, warp_matrix):
        img = results['img']

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def _transform_bboxes(self, results, width, height, warp_matrix, scaling_ratio):

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack(
                    (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = \
                        warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = \
                        warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes)
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ['gt_bboxes']:
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]

                if 'gt_masks' in results:
                    raise NotImplementedError(
                        'RandomAffine only supports bbox.')

        return results

    def _get_affine_param(self, height, width):
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
                translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix, scaling_ratio


@PIPELINES.register_module()
class SeqMixUp(MixUp):
    """Sequence MixUp augmentation.

    This operation uses the same augmentation parameters for all images in the sequence
    """
    def __call__(self, results):
        random_params = None
        outs = []
        for frame_idx, _results in enumerate(results):
            _results, random_params = self._mixup_transform(_results, random_params)
            outs.append(_results)
        return outs

    def _mixup_transform(self, results, random_params=None):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # calculate random augmentation parameters
        jit_factor, is_flip, y_offset, x_offset = None, None, None, None
        if random_params is not None:
            jit_factor, is_flip, y_offset, x_offset = random_params

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results, random_params

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = jit_factor or random.uniform(*self.ratio_range)
        is_flip = is_flip or random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        if padded_img.shape[0] > target_h:
            y_offset = y_offset or random.randint(0, padded_img.shape[0] - target_h)
        else:
            y_offset = 0
        if padded_img.shape[1] > target_w:
            x_offset = x_offset or random.randint(0, padded_img.shape[1] - target_w)
        else:
            x_offset = 0
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2],
                                                  0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2],
                                                  0, origin_h)

        if is_flip:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = \
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = \
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2], 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2], 0, target_h)

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                    cp_retrieve_gt_bboxes.T)

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate(
            (results['gt_labels'], retrieve_gt_labels), axis=0)

        # remove outside bbox
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels

        random_params = (jit_factor, is_flip, y_offset, x_offset)
        return results, random_params


@PIPELINES.register_module()
class SeqYOLOXHSVRandomAug(YOLOXHSVRandomAug):
    """Sequence HSV augmentation.

    This operation uses the same augmentation parameters for all images in the sequence
    """
    def __call__(self, results):
        outs = []
        hsv_gains = self._get_hsv_gains()
        for _results in results:
            _results = self._apply_hsv_gains(_results, hsv_gains)
            outs.append(_results)
        return outs

    def _apply_hsv_gains(self, results, hsv_gains):
        img = results['img']

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results['img'] = img
        return results

    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains


@PIPELINES.register_module()
class SeqCustomFormatBundle(SeqDefaultFormatBundle):
    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self.default_format_bundle(_results)
            outs.append(_results)
        data = {}
        data['num_frames'] = len(outs)
        for i, out in enumerate(outs):
            for k, v in out.items():
                if i == 0:
                    data[k] = v
                else:
                    data[f'{k}_{i}'] = v
        return data


@PIPELINES.register_module()
class SeqFilterAnnotations(FilterAnnotations):

    def __call__(self, results):
        return [super(SeqFilterAnnotations, self).__call__(_r) for _r in results]


@PIPELINES.register_module()
class SeqCollect(object):
    def __init__(self,
                 keys,
                 meta_keys=None,
                 default_meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor',
                                    'flip', 'flip_direction', 'img_norm_cfg',
                                    'frame_id', 'is_video_data',
                                    )):
        self.keys = keys
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def __call__(self, results):
        num_frames = results['num_frames']
        outs = dict(num_frames=num_frames)
        for i in range(num_frames):
            _results = self._collect_meta_keys(results, i)
            outs.update(_results)

        return outs

    def _collect_meta_keys(self, results, frame_idx):
        """Collect `self.keys` and `self.meta_keys` from `results` (dict)."""
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            frame_key = f'{key}_{frame_idx}' if frame_idx > 0 else key
            frame_info_key = f'img_info_{frame_idx}' if frame_idx > 0 else 'img_info'
            if frame_key in results:
                img_meta[key] = results[frame_key]
            elif key in results[frame_info_key]:
                img_meta[key] = results[frame_info_key][key]
        img_meta_key = f'img_metas_{frame_idx}' if frame_idx > 0 else 'img_metas'
        data[img_meta_key] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            frame_key = f'{key}_{frame_idx}' if frame_idx > 0 else key
            data[frame_key] = results[frame_key]
        return data


@PIPELINES.register_module()
class SeqMultiScaleFlipAug(MultiScaleFlipAug):
    def __call__(self, results):
        return [super(SeqMultiScaleFlipAug, self).__call__(_r) for _r in results]