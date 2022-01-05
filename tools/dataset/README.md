# Data preparation
## Download dataset and ckpt
Use flags depending on which datasets to download:
`python download_extract_data.py --moe #--faces --full --icf_train --icf_test --ckpt_sketch --ckpt_aoda`

`mv model.pth ckpt`

## Prepare raw (RGB) data for paired or unpaired translation
Use `prepare_dataset.py` to make data into paired or unpaired folders 
compatible with MMGen. Example:

`python --path_images /ABSOLUTE/PATH/DATA/moe/ --path_save /ABSOLUTE/PATH/DATA/paired/moe/ --preprocess METHOD`

Full list of arguments:
```
usage: prepare_dataset.py [-h] [--path_images PATH_IMAGES] [--path_save PATH_SAVE] [--print_freq PRINT_FREQ]
                          [--unpaired] [--preprocess {linear}] [--train TRAIN] [--res_hw RES_HW]

optional arguments:
  -h, --help            show this help message and exit
  --path_images PATH_IMAGES
                        folder with images to convert
  --path_save PATH_SAVE
                        folder to save new paired images
  --print_freq PRINT_FREQ
                        printfreq
  --unpaired            Use for unpaired (cycleGAN style) datasets
  --preprocess {linear}
                        greyscale/sketch conversion
  --train TRAIN         percent of data for training

```

## Symbolic link from data path to expected by configs
`ln -s /ABSOLUTE/PATH/DATA/paired/moe/ /ABSOLUTE/PATH/MMGEN/data/paired/moe/`
