import os
import os.path as osp
import argparse
import glob

import numpy as np
from PIL import Image
from skimage.transform import resize

from transformations import convert_image


def search_images(args):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = osp.join(args.path_images, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files: ', len(files_all))
    return files_all


def read_image(fp):
    img = Image.open(fp)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def get_save_folder_name(i, no_train, args, sub_folder_name):
    if args.unpaired:
        if i < no_train:
            folder_name = osp.join(args.path_save, 'trainDOM', sub_folder_name)
        else:
            folder_name = osp.join(args.path_save, 'testDOM', sub_folder_name)
    else:
        if i < no_train:
            folder_name = osp.join(args.path_save, 'train', sub_folder_name)
        else:
            folder_name = osp.join(args.path_save, 'val', sub_folder_name)

    return folder_name


def save_image(args, img, img_new, folder_name, fn):
    if args.unpaired:
        save_image_unpaired(args.res_hw, img, img_new, folder_name, fn)
    else:
        save_image_paired(args.res_hw, img, img_new, folder_name, fn)


def save_image_paired(res_hw, img, img_new, folder_name, fn):
    img = resize(np.array(img), (res_hw, res_hw))

    img_new = np.stack((img_new,)*3, axis=-1)
    img_new = resize(img_new, (res_hw, res_hw))

    img_combined = np.hstack((img_new, img)) * 255
    img_combined = Image.fromarray(img_combined.astype('uint8'), 'RGB')

    os.makedirs(osp.join(folder_name), exist_ok=True)
    new_fn = osp.join(folder_name, fn)
    img_combined.save(new_fn)


def save_image_unpaired(res_hw, img, img_new, folder_name, fn):
    img_new = resize(np.array(img_new), (res_hw, res_hw)) * 255
    img_new = Image.fromarray(img_new.astype('uint8'), 'L')

    new_folder_name = folder_name.replace('DOM', 'A')
    os.makedirs(osp.join(new_folder_name), exist_ok=True)
    new_fn = osp.join(new_folder_name, fn)
    img_new.save(new_fn)

    img = resize(np.array(img), (res_hw, res_hw)) * 255
    img = Image.fromarray(img.astype('uint8'), 'RGB')

    new_folder_name = folder_name.replace('DOM', 'B')
    os.makedirs(osp.join(new_folder_name), exist_ok=True)
    new_fn = osp.join(new_folder_name, fn)
    img.save(new_fn)


def preprocess_folder(args):
    files_all = search_images(args)
    no_images = len(files_all)
    no_train = int(no_images * args.train)
    no_val = no_images - no_train

    for i, fp in enumerate(files_all):
        abs_path, fn = osp.split(osp.normpath(fp))
        _, sub_folder_name = osp.split(abs_path)
        img = read_image(fp)
        img_new = convert_image(args, img)

        folder_name = get_save_folder_name(i, no_train, args, sub_folder_name)
        save_image(args, img, img_new, folder_name, fn)

        if i % args.print_freq == 0:
            print(f'{i}/{len(files_all)}: {fp}')

    print(f'''Finished converting images. Results saved in {args.path_save}.
        Total images: {no_images} Training: {no_train} Val: {no_val}''')


def main():
    '''takes a folder with images and applies a transformation to all images'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_images', type=str,
                        help='folder with images to convert')
    parser.add_argument('--path_save', type=str,
                        help='folder to save new paired images')
    parser.add_argument('--print_freq',
                        type=int, default=1000, help='printfreq')
    parser.add_argument('--unpaired', action='store_true',
                        help='Use for unpaired (cycleGAN style) datasets')
    parser.add_argument('--preprocess', type=str, choices=['linear'],
                        default='linear', help='greyscale/sketch conversion')
    parser.add_argument('--train', type=float, default=0.99,
                        help='percent of data for training')
    parser.add_argument('--res_hw', type=int, default=256,
                        help='resized height and width')
    args = parser.parse_args()

    preprocess_folder(args)


main()
