# links with the models and related files
https://github.com/arkel23/mmgeneration/blob/master/tools/dataset/models_aoda/aoda.py
https://github.com/arkel23/mmgeneration/blob/master/tools/dataset/models_sketchkeras/sketchkeras.py
https://github.com/arkel23/mmgeneration/blob/master/tools/dataset/build_aoda.py
https://github.com/arkel23/mmgeneration/blob/master/tools/dataset/build_sketchkeras.py

# install
pip install gdown

# download ckpts
gdown 1Zo88NmWoAitO7DnyBrRhKXPcHyMAZS97 -O ckpts/model_sketch.pth
gdown 1RILKwUdjjBBngB17JHwhZNBEaW4Mr-Ml -O ckpts/model_aoda.pth

# try sketchkeras and aoda  to verify the network was downloaded and it can load normally
python build_sketchkeras.py
python build_aoda.py

# preprocess samples / dataset --path_images points to folder with images to process
# --path_save to where to save the images
# --preprocess aoda for aoda or --preprocess sketchkeras
# unpaired to save only the results (otherwise it saves both the original and processed side by side)
python -u prepare_dataset.py --path_images movies/ --path_save results/data_aoda --preprocess aoda --unpaired

