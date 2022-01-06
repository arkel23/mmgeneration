_base_ = ['pix2pix_moe_linear.py']

dataroot = 'data/paired/moe_canny'
data = dict(train=dict(dataroot=dataroot))

# runtime settings
exp_name = 'pix2pix_moe_canny'
work_dir = f'./work_dirs/experiments/{exp_name}'
