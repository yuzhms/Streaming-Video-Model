# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS, CocoDataset
from terminaltables import AsciiTable

from mmtrack.core import eval_mot
from mmtrack.utils import get_root_logger
from ..parsers import CocoVID


@DATASETS.register_module()
class CustomCOCOVideo(CocoDataset):
    """Base coco video dataset for VID, MOT and SOT tasks.

    Args:
        ref_img_sampler (dict): Configuration of sampling ref images.
        test_load_ann (bool): If True, loading annotations during testing,
            otherwise, not loading. Default: False.
    """

    CLASSES = ('pedestrian', )

    def __init__(self,
                 sample_ratio=1.,
                 ref_img_sampler=dict(
                     stride=1, num_frames=16),
                 test_load_ann=False,
                 is_source_image=False,
                 *args,
                 **kwargs):
        assert ref_img_sampler is not None
        self.load_as_video = True
        self.ref_img_sampler = ref_img_sampler
        self.sample_ratio = sample_ratio
        self.test_load_ann = test_load_ann
        self.is_source_image = is_source_image
        super().__init__(*args, **kwargs)
        self.logger = get_root_logger()

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """
        data_infos = self.load_video_anns(ann_file)
        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self._load_coco_ann(ann_file)
        data_infos = self._create_data_infos()
        return data_infos

    def _load_coco_ann(self, ann_file):
        self.coco = CocoVID(ann_file, load_img_as_vid=self.is_source_image)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def _create_data_infos(self):
        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.sample_ratio < 1.:
                img_ids = sorted(random.sample(img_ids, k=int(len(img_ids) * self.sample_ratio)))
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def resample(self):
        data_infos = self._create_data_infos()
        self.data_infos = data_infos

    def get_ann_info(self, img_info):
        """Get COCO annotations by the information of image.

        Args:
            img_info (dict|int): Information of image.

        Returns:
            dict: Annotation information of `img_info`.
        """
        if isinstance(img_info, int):
            img_info = self.data_infos[img_info]
        img_id = img_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(img_info, ann_info)

    def prepare_results(self, img_info):
        """Prepare results for image (e.g. the annotation information, ...)."""
        results = dict(img_info=img_info)
        if not self.test_mode or self.test_load_ann:
            results['ann_info'] = self.get_ann_info(img_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info['id'])
            results['proposals'] = self.proposals[idx]

        super().pre_pipeline(results)
        results['is_video_data'] = self.load_as_video
        return results

    def ref_img_sampling(self,
                         img_info,
                         stride=1,
                         num_frames=16):
        vid_id, img_id, frame_id = img_info['video_id'], img_info['id'], img_info['frame_id']
        img_ids = self.coco.get_img_ids_from_vid(vid_id)

        if isinstance(stride, (tuple, list)):
            stride = random.choice(stride)

        if num_frames == -1:
            num_frames = len(img_ids)
            assert stride == 1

        ref_img_infos = [img_info]
        for i in range(1, num_frames):
            ref_frame_idx = min(frame_id + i * stride, len(img_ids) - 1)
            ref_img_idx = img_ids[ref_frame_idx]
            ref_img_info = self.coco.load_imgs([ref_img_idx])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
            ref_img_infos.append(ref_img_info)
        return ref_img_infos

    def prepare_data(self, idx):
        """Get data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            list[dict]: Data and annotations after pipeline with new keys introduced
            by pipeline.
        """
        img_info = self.data_infos[idx]
        img_infos = self.ref_img_sampling(img_info, **self.ref_img_sampler)
        results = [
            self.prepare_results(img_info) for img_info in img_infos
        ]

        return self.pipeline(results)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotations after pipeline with new keys
            introduced by pipeline.
        """
        return self.prepare_data(idx)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
            pipeline.
        """
        return self.prepare_data(idx)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotations.

        Args:
            img_anfo (dict): Information of image.
            ann_info (list[dict]): Annotation information of image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
            labels, instance_ids, masks, seg_map. "masks" are raw
            annotations and not decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_instance_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if 'segmentation' in ann:
                    gt_masks.append(ann['segmentation'])
                if 'instance_id' in ann:
                    gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks,
            seg_map=seg_map)

        ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)

        return ann

    def evaluate(self,
                 results,
                 metric=['bbox', 'track'],
                 logger=None,
                 bbox_kwargs=dict(
                     classwise=False,
                     proposal_nums=(100, 300, 1000),
                     iou_thrs=None,
                     metric_items=None),
                 track_kwargs=dict(
                     iou_thr=0.5,
                     ignore_iof_thr=0.5,
                     ignore_by_classes=False,
                     nproc=4)):
        """Evaluation in COCO protocol and CLEAR MOT metric (e.g. MOTA, IDF1).

        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            bbox_kwargs (dict): Configuration for COCO styple evaluation.
            track_kwargs (dict): Configuration for CLEAR MOT evaluation.

        Returns:
            dict[str, float]: COCO style and CLEAR MOT evaluation metric.
        """
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['bbox', 'segm', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        eval_results = dict()
        if 'track' in metrics:
            assert len(self.data_infos) == len(results['track_bboxes'])
            inds = [
                i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0
            ]
            num_vids = len(inds)
            inds.append(len(self.data_infos))

            track_bboxes = [
                results['track_bboxes'][inds[i]:inds[i + 1]]
                for i in range(num_vids)
            ]
            ann_infos = [self.get_ann_info(_) for _ in self.data_infos]
            ann_infos = [
                ann_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
            ]
            track_eval_results = eval_mot(
                results=track_bboxes,
                annotations=ann_infos,
                logger=logger,
                classes=self.CLASSES,
                **track_kwargs)
            eval_results.update(track_eval_results)

        # evaluate for detectors without tracker
        super_metrics = ['bbox', 'segm']
        super_metrics = [_ for _ in metrics if _ in super_metrics]
        if super_metrics:
            if isinstance(results, dict):
                if 'bbox' in super_metrics and 'segm' in super_metrics:
                    super_results = []
                    for bbox, mask in zip(results['det_bboxes'],
                                          results['det_masks']):
                        super_results.append((bbox, mask))
                else:
                    super_results = results['det_bboxes']
            elif isinstance(results, list):
                super_results = results
            else:
                raise TypeError('Results must be a dict or a list.')
            super_eval_results = super().evaluate(
                results=super_results,
                metric=super_metrics,
                logger=logger,
                **bbox_kwargs)
            eval_results.update(super_eval_results)

        return eval_results

    def __repr__(self):
        return ''