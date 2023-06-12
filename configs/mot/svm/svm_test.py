_base_ = [
    '../../_base_/models/yolox_x_vit.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet_ext'],
    allow_failed_imports=False)

img_scale = (800, 1440)
samples_per_gpu = 2
num_frames = 32
train_sample_stride = (1, 2, 3, 4)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='ByteTrack',
    detector=dict(
        backbone=dict(num_frames=num_frames,
                      residual_block_index=(2, 5, 8, 11),
                      type='SViT',
                      attn_type='stream',
                      enable_checkpoint=True),
        random_size_interval=num_frames,
        input_size=img_scale,
        random_size_range=(18, 32),
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7))),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.7, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

train_pipeline = [
    dict(
        type='SeqMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='SeqRandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='SeqMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='SeqYOLOXHSVRandomAug'),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqResize',
        share_params=True,
        img_scale=img_scale,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32, pad_val=dict(img=(0., 0., 0.))),
    dict(type='SeqFilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='SeqCustomFormatBundle'),
    dict(type='SeqCollect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(
        type='SeqMultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(0., 0., 0.))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

MOTSynth_dataset = dict(
    type='CustomCOCOVideo',
    ref_img_sampler=dict(num_frames=num_frames, stride=train_sample_stride),
    sample_ratio=0.1,
    ann_file='data/MOTSynth/all_cocoformat.json',
    img_prefix='data/MOTSynth/train',
    classes=('pedestrian',),
    pipeline=[
        dict(type='LoadMultiImagesFromFile'),
        dict(type='SeqLoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False
)

MOT17_dataset = dict(
    type='CustomCOCOVideo',
    ref_img_sampler=dict(num_frames=num_frames, stride=train_sample_stride),
    sample_ratio=1.,
    ann_file='data/MOT17/annotations/half-train_cocoformat.json',
    # ann_file='data/MOT17/annotations/train_cocoformat.json',
    img_prefix='data/MOT17/train',
    classes=('pedestrian',),
    pipeline=[
        dict(type='LoadMultiImagesFromFile'),
        dict(type='SeqLoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False
)

CrowdHuman_dataset_train = dict(
    type='CustomCOCOVideo',
    ref_img_sampler=dict(num_frames=num_frames, stride=1),
    is_source_image=True,
    sample_ratio=1.,
    ann_file='data/crowdhuman/annotations/crowdhuman_train.json',
    img_prefix='data/crowdhuman/train',
    classes=('pedestrian',),
    pipeline=[
        dict(type='LoadMultiImagesFromFile'),
        dict(type='SeqLoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False
)

CrowdHuman_dataset_val = dict(
    type='CustomCOCOVideo',
    ref_img_sampler=dict(num_frames=num_frames, stride=1),
    is_source_image=True,
    sample_ratio=1.,
    ann_file='data/crowdhuman/annotations/crowdhuman_val.json',
    img_prefix='data/crowdhuman/val',
    classes=('pedestrian',),
    pipeline=[
        dict(type='LoadMultiImagesFromFile'),
        dict(type='SeqLoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=1,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='MultiVideoMixDataset',
        dataset=[MOTSynth_dataset, MOT17_dataset, CrowdHuman_dataset_train, CrowdHuman_dataset_val],
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type='MOTChallengeDatasetVideo',
        ann_file='data/MOT17/annotations/half-val_cocoformat.json',
        img_prefix='data/MOT17/train',
        ref_img_sampler=dict(stride=1, num_frames=-1),
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20),
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type='MOTChallengeDatasetVideo',
        ann_file='data/MOT17/annotations/half-val_cocoformat.json',
        img_prefix='data/MOT17/train',
        ref_img_sampler=dict(stride=1, num_frames=-1),
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20),
        pipeline=test_pipeline),
)

# optimizer
# default 8 gpu
optimizer = dict(
    _delete_=True,
    constructor='CustomOptimizerConstructor',
    type='AdamW',
    lr=0.00025, # 0.001 / 8 * samples_per_gpu,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0,
                       custom_keys={
                           'residual_block': dict(lr_mult=1.0),
                           'backbone.fpn_adaptor': dict(lr_mult=1.0),
                           'backbone.ln_post_': dict(lr_mult=1.0),
                           # '_t': dict(lr_mult=1.0),
                           # 'alpha': dict(lr_mult=1.0),
                           'backbone': dict(lr_mult=0.1),
                       }))
optimizer_config = dict(grad_clip=None)

# some hyper parameters
total_epochs = 10
num_last_epochs = 1
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1 * num_frames,  # should be fixed by multiplying num_frames
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

custom_hooks = [
    # dict(
    #     type='YOLOXModeSwitchHook',
    #     num_last_epochs=num_last_epochs,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49),
    dict(type='DataResampleHook', interval=1, priority=91),
]

checkpoint_config = dict(interval=1, max_keep_ckpts=10)
evaluation = dict(metric=['bbox', 'track'], interval=1, save_best='HOTA', gpu_collect=False)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))

seq_runner = True
