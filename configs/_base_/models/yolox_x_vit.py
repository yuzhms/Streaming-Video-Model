# model settings
img_scale = (640, 640)

custom_imports = dict(
    imports=['mmdet_ext'],
    allow_failed_imports=False)

model = dict(
    detector=dict(
        type='YOLOX',
        input_size=img_scale,
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(
            type='ViT',
            width=768,
            patch_size=(16, 16),
            layers=12,
            heads=12,
            out_indices=(1, 2, 3),
            window_size=14,
            window_block_index=list(range(12)),
        ),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            num_csp_blocks=4,
            norm_cfg=dict(type='SyncBN', momentum=0.03, eps=0.001),
        ),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=80,
            in_channels=256,
            feat_channels=256,
            norm_cfg=dict(type='SyncBN', momentum=0.03, eps=0.001),
        ),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))))
