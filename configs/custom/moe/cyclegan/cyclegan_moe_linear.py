_base_ = [
    '../_base_/models/cyclegan/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]

domain_a = 'bw'
domain_b = 'color'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

model = dict(
    default_domain=domain_b,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_b}',
                target=f'real_{domain_b}',
            ),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_b}', target=f'real_{domain_b}'),
            reduction='mean')
    ])
dataroot = './data/unpaired/moe_linear'
dataroot_test = 'data/paired/test_daf_faces_0000'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=[f'img_{domain_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{domain_b}'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

inference_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_a}'],
        to_rgb=False,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}'],
        meta_keys=[f'img_{domain_a}_path'])
]

data = dict(
    train=dict(
        dataroot=dataroot,
        pipeline=train_pipeline,
        domain_a=domain_a,
        domain_b=domain_b),
    val=dict(
        dataroot=dataroot_test,
        domain_a=domain_a,
        domain_b=domain_b,
        pipeline=test_pipeline),
    test=dict(
        dataroot=dataroot_test,
        domain_a=domain_a,
        domain_b=domain_b,
        pipeline=test_pipeline))

optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=135000, interval=1350)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=[f'fake_{domain_a}', f'fake_{domain_b}'],
        interval=5000)
]

runner = None
use_ddp_wrapper = True
total_iters = 270000
workflow = [('train', 1)]
exp_name = 'cyclegan_moe_linear'
work_dir = f'./work_dirs/experiments/{exp_name}'
# testA 120, testB 140
num_images = 100
metrics = dict(
    FID=dict(type='FID', num_images=num_images, image_shape=(3, 256, 256)),
    IS=dict(
        type='IS',
        num_images=num_images,
        image_shape=(3, 256, 256),
        inception_args=dict(type='pytorch')))

evaluation = dict(
    type='TranslationEvalHook',
    target_domain=domain_b,
    interval=10000,
    metrics=[
        dict(type='FID', num_images=num_images, bgr2rgb=True),
        dict(
            type='IS',
            num_images=num_images,
            inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])

# log
log_config = dict(
    by_epoch=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook')
    ])
