# dataset settings
dataset_type = 'BrickKilnDataset'
data_root = '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/data/thera_rdn_pro/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(2560, 2560), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(2560, 2560), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(2560, 2560), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='bihar_4x/annfiles/',
        data_prefix=dict(img_path='bihar_4x/images_png/'),
        # ann_file = 'haryana_4x/annfiles/',
        # data_prefix = dict(img_path='haryana_4x/images_png/'),
        img_shape=(2560, 2560),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_bihar_4x/annfiles/',
        data_prefix=dict(img_path='test_bihar_4x/images_png/'),
        img_shape=(2560, 2560),
        test_mode=True,
        pipeline=val_pipeline))
# test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
# test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        img_shape=(2560, 2560),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(type='DOTAMetric', format_only=True, merge_patches=True)
