_base_ = ['pix2pix_moe_linear.py']

dataroot = 'data/paired/moe_sketchkeras'
data = dict(train=dict(dataroot=dataroot))

# runtime settings
exp_name = 'pix2pix_moe_sketchkeras'
work_dir = f'./work_dirs/experiments/{exp_name}'
