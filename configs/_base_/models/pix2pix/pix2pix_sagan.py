source_domain = None  # set by user
target_domain = None  # set by user
init_cfg = dict(type='studio')
# model settings
model = dict(
    type='Pix2Pix',
    generator=dict(
        type='SAGANGenerator',
        output_scale=128,
        base_channels=64,
        attention_cfg=dict(type='SelfAttentionBlock'),
        attention_after_nth_block=4,
        with_spectral_norm=True,
        use_cbn=False,
        # num_classes=1000,
        init_cfg=init_cfg),
    discriminator=dict(
        type='ProjDiscriminator',
        input_scale=128,
        base_channels=64,
        attention_cfg=dict(type='SelfAttentionBlock'),
        attention_after_nth_block=1,
        with_spectral_norm=True,
        # use_cbn=False,
        # num_classes=1000,
        init_cfg=init_cfg),
    gan_loss=dict(
        type='GANLoss',
        gan_type='hinge',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain=target_domain,
    reachable_domains=[target_domain],
    related_domains=[target_domain, source_domain],
    gen_auxiliary_loss=dict(
        type='L1Loss',
        loss_weight=100.0,
        loss_name='pixel_loss',
        data_info=dict(
            pred=f'fake_{target_domain}', target=f'real_{target_domain}'),
        reduction='mean'))
# model training and testing settings
train_cfg = dict(disc_steps=1)
test_cfg = None
