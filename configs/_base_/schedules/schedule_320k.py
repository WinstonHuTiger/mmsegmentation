# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=320000,
        by_epoch=False)
]
# training schedule for 320k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=32000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=32000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
