import time
import torch
from mmcv.runner import EpochBasedRunner
from mmcv.parallel import is_module_wrapper

def split_data_batch(data_batch):
    # img = data_batch.pop('img')
    num_frames = data_batch.pop('num_frames')[0]

    data_batch_list = []
    for i in range(num_frames):
        img_key = f'img_{i}' if i > 0 else 'img'
        meta_key = f'img_metas_{i}' if i > 0 else 'img_metas'
        box_key = f'gt_bboxes_{i}' if i > 0 else 'gt_bboxes'
        label_key = f'gt_labels_{i}' if i > 0 else 'gt_labels'
        _data_batch = dict(img=data_batch.pop(img_key),  # [:, i]
                           img_metas=data_batch.pop(meta_key),
                           gt_bboxes=data_batch.pop(box_key),
                           gt_labels=data_batch.pop(label_key))
        data_batch_list.append(_data_batch)
    return data_batch_list


def clear_history(runner):
    model = runner.model
    if is_module_wrapper(model):
        model = model.module
    if hasattr(model, 'detector') and \
       hasattr(model.detector, 'backbone') and \
       hasattr(model.detector.backbone, 'clear_history'):
        model.detector.backbone.clear_history()


class SeqEpochBasedRunner(EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader) * self.meta.get('num_frames', 1)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        clear_history(self)
        for i, data_batch in enumerate(self.data_loader):
            data_batch_list = split_data_batch(data_batch)
            for frame_idx, _data_batch in enumerate(data_batch_list):
                self.data_batch = _data_batch
                self._inner_iter = i * self.meta.get('num_frames', 1) + frame_idx
                self.call_hook('before_train_iter')
                self.run_iter(_data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                del self.data_batch
                self._iter += 1
            clear_history(self)

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        raise NotImplementedError
