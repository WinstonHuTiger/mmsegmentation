# # dataset settings
# dataset_type1 = 'COCOStuffDataset'

# dataset_type2 = "pseudoLabelDataset"

# data_root1 = '/dev_data/data/stego_dataset/cocostuff'

# data_root2 = "/dev_data/dev/unsupervised_segmentation_diffusion/usd/new_synthesized_images"
# crop_size = (512, 512)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(
#         type='RandomResize',
#         scale=(2048, 512),
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2048, 512), keep_ratio=True),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type='LoadAnnotations'),
#     dict(type='PackSegInputs')
# ]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
# train_dataloader = dict(
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type2,
#         data_root=data_root2,
#         data_prefix=dict(
#             img_path='images', seg_map_path='labels'),
#         pipeline=train_pipeline))

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type1,
#         data_root=data_root1,
#         data_prefix=dict(
#             img_path='images/val2017', seg_map_path='annotations/val2017'),
#         pipeline=test_pipeline))
# test_dataloader = val_dataloader

# val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = val_evaluator

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/dev_data/data/stego_dataset/VOCdevkit/VOC2012'
# dataset_type1 = 'COCOStuffDataset'

dataset_type2 = "pseudoLabelDataset"

# data_root1 = '/dev_data/data/stego_dataset/cocostuff'

data_root2 = "/dev_data/dev/unsupervised_segmentation_diffusion/usd/voc_simple_synthesized"

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type2,
        data_root=data_root2,
        data_prefix=dict(
            img_path='images', seg_map_path='labels'),
        pipeline=train_pipeline))



val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
