# linear,canny,xdog,xdog_th,xdog_serial,sketchkeras,aoda,pidinet_-1,pidinet_0,pidinet_1,pidinet_2,pidinet_3
# --train 0.99 --res_hw 256 paired --print_freq 1000
nohup python -u prepare_dataset.py --path_images /edahome/pcslab/pcs05/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/faces/0000 --path_save ../../../../data/paired/test_daf_faces_0000 --preprocess linear canny xdog xdog_th xdog_serial_0.3 xdog_serial_0.4 xdog_serial_0.5 sketchkeras aoda pidinet_-1 pidinet_0 pidinet_1 pidinet_2 pidinet_3 --train 0.01

nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_linear --preprocess linear
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_canny --preprocess canny
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_xdog --preprocess xdog
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_xdog_th --preprocess xdog_th
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_xdog_serial0.3 --preprocess xdog_serial_0.3
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_xdog_serial0.4 --preprocess xdog_serial_0.4
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_xdog_serial0.5 --preprocess xdog_serial_0.5

nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_sketchkeras --preprocess sketchkeras
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_aoda --preprocess aoda
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_pidinet_-1 --preprocess pidinet_-1
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_pidinet_0 --preprocess pidinet_0
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_pidinet_1 --preprocess pidinet_1
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_pidinet_2 --preprocess pidinet_2
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_pidinet_3 --preprocess pidinet_3

nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_trad --preprocess linear canny xdog xdog_th xdog_serial_0.3 xdog_serial_0.4 xdog_serial_0.5
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_dl --preprocess sketchkeras aoda pidinet_-1 pidinet_0 pidinet_1 pidinet_2 pidinet_3
nohup python -u prepare_dataset.py --path_images ../../../../data/moeimouto_animefacecharacterdataset/data/ --path_save ../../../../data/paired/moe_all --preprocess linear canny xdog xdog_th xdog_serial_0.3 xdog_serial_0.4 xdog_serial_0.5 sketchkeras aoda pidinet_-1 pidinet_0 pidinet_1 pidinet_2 pidinet_3
