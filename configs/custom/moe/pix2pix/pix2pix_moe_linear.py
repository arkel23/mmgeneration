_base_ = [
    '../../../_base_/models/pix2pix/pix2pix_vanilla_unet_bn.py',
    '../../../_base_/datasets/paired_imgs_256x256.py',
    '../../../_base_/default_runtime.py'
]
source_domain = 'bw'
target_domain = 'color'
# model settings
model = dict(
    default_domain=target_domain,
    reachable_domains=[target_domain],
    related_domains=[target_domain, source_domain],
    gen_auxiliary_loss=dict(
        data_info=dict(
            pred=f'fake_{target_domain}', target=f'real_{target_domain}')))
# dataset settings
domain_a = source_domain
domain_b = target_domain
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='FixedCrop',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        crop_size=(256, 256)),
    dict(
        type='Flip',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        to_rgb=False,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
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
        **img_norm_cfg),
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

dataroot = 'data/paired/moe_linear'
dataroot_test = 'data/paired/test_daf_faces_0000'
data = dict(
    train=dict(dataroot=dataroot, pipeline=train_pipeline),
    val=dict(dataroot=dataroot_test, pipeline=test_pipeline, testdir='val'),
    test=dict(dataroot=dataroot_test, pipeline=test_pipeline, testdir='val'))

# optimizer
optimizer = dict(
    generators=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = None

# checkpoint saving
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=[f'fake_{target_domain}'],
        interval=5000)
]
runner = None
use_ddp_wrapper = True

# runtime settings
total_iters = 190000
workflow = [('train', 1)]
exp_name = 'pix2pix_moe_linear'
work_dir = f'./work_dirs/experiments/{exp_name}'
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
