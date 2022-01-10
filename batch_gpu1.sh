# pix2pix
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_sketchkeras.py
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_aoda.py
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_pidinet_-1.py
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_pidinet_0.py
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_pidinet_1.py
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_pidinet_2.py
#CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_pidinet_3.py

#nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_trad.py
#nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_dl.py
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_all.py

# cyclegan
#nohup python -u tools/train.py configs/custom/cyclegan_id0_moe_linear.py
#nohup python -u tools/train.py configs/custom/cyclegan_moe_linear.py
