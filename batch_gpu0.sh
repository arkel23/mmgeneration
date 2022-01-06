# pix2pix
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_linear.py
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_canny.py
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_xdog.py
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_xdog_th.py
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_xdog_serial_03.py
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_xdog_serial_04.py
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_xdog_serial_05.py

#nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_trad.py
#nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_dl.py
#nohup python -u tools/train.py configs/custom/moe/pix2pix/pix2pix_moe_all.py

# cyclegan
#nohup python -u tools/train.py configs/custom/cyclegan_id0_moe_linear.py
#nohup python -u tools/train.py configs/custom/cyclegan_moe_linear.py
