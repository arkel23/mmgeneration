_base_ = ['pix2pix_moe_linear.py']

dataroot = 'data/paired/moe_pidinet_2'
data = dict(train=dict(dataroot=dataroot))

# runtime settings
exp_name = 'pix2pix_moe_pidinet_2'
work_dir = f'./work_dirs/experiments/{exp_name}'
