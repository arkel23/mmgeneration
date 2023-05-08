import os
import os.path as osp
import argparse
import glob
import copy

import numpy as np
from PIL import Image
from skimage.transform import resize

from transformations import convert_image
from build_sketchkeras import get_sketchkeras
from build_aoda import get_aoda
from build_pidinet import get_pidinet


ALGOS_TRAD = ['linear', 'canny', 'xdog', 'xdog_th', 'xdog_serial_0.3',
              'xdog_serial_0.4', 'xdog_serial_0.5']
ALGOS_DL = ['sketchkeras', 'aoda',
            'pidinet_-1', 'pidinet_0', 'pidinet_1', 'pidinet_2', 'pidinet_3']
ALGOS = ALGOS_TRAD + ALGOS_DL


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


def get_model(args):

    if isinstance(args.preprocess, str):
        if args.preprocess in ALGOS_DL:
            if args.preprocess == 'sketchkeras':
                model = get_sketchkeras()
            elif args.preprocess == 'aoda':
                model = get_aoda()
            elif 'pidinet' in args.preprocess:
                model = get_pidinet()
        else:
            model = None
    elif isinstance(args.preprocess, list):
        model = []
        for alg in args.preprocess:
            if alg in ALGOS_DL:
                if alg == 'sketchkeras':
                    model.append(get_sketchkeras())
                elif alg == 'aoda':
                    model.append(get_aoda())
                elif 'pidinet' in alg:
                    model.append(get_pidinet())
            else:
                model.append(None)

    return model


def read_image(fp):
    img = Image.open(fp)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def convert_save_single(args, img, model, i, no_train, sub_folder_name, fn):
    img_new = convert_image(args, img, model)

    folder_name = get_save_folder_name(i, no_train, args, sub_folder_name)
    save_image(args, img, img_new, folder_name, fn)


def get_save_folder_name(i, no_train, args, sub_folder_name):
    sub_folder_name = f'{sub_folder_name}_{args.preprocess}'

    if args.save_processed_only:
        folder_name = osp.join(args.path_save, sub_folder_name)
    elif args.unpaired:
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
    if args.save_processed_only:
        save_image_processed_only(args.res_hw, img_new, folder_name, fn)  
    elif args.unpaired:
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


def save_image_processed_only(res_hw, img_new, folder_name, fn):
    img_new = resize(np.array(img_new), (res_hw, res_hw)) * 255
    img_new = Image.fromarray(img_new.astype('uint8'), 'L')

    os.makedirs(osp.join(folder_name), exist_ok=True)
    new_fn = osp.join(folder_name, fn)
    img_new.save(new_fn)


def preprocess_folder(args):
    files_all = search_images(args)
    no_images = len(files_all)
    no_train = int(no_images * args.train)
    no_val = no_images - no_train

    model = get_model(args)

    for i, fp in enumerate(files_all):
        abs_path, fn = osp.split(osp.normpath(fp))
        _, sub_folder_name = osp.split(abs_path)
        img = read_image(fp)

        if isinstance(args.preprocess, str):
            convert_save_single(
                args, img, model, i, no_train, sub_folder_name, fn)
        else:
            for j, alg in enumerate(args.preprocess):
                args_copy = copy.deepcopy(args)
                args_copy.preprocess = alg
                convert_save_single(
                    args_copy, img, model[j], i, no_train, sub_folder_name, fn)

        if i % args.print_freq == 0:
            print(f'{i}/{len(files_all)}: {fp}')

    print(f'''Finished converting images. Results saved in {args.path_save}.
        Total images: {no_images} Training: {no_train} Val: {no_val}''')


def main():
    '''takes a folder with images and applies a transformation to all images'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_images', type=str, required=True,
                        help='folder with images to convert')
    parser.add_argument('--path_save', type=str, required=True,
                        help='folder to save new paired images')
    parser.add_argument('--print_freq',
                        type=int, default=1000, help='printfreq')

    parser.add_argument('--save_processed_only', action='store_true',
                        help='save processed images only')
    parser.add_argument('--unpaired', action='store_true',
                        help='Use for unpaired (cycleGAN style) datasets')
    parser.add_argument('--preprocess', nargs='+', type=str, choices=ALGOS,
                        default='linear', help='greyscale/sketch conversion')

    parser.add_argument('--train', type=float, default=0.99,
                        help='percent of data for training')
    parser.add_argument('--res_hw', type=int, default=256,
                        help='resized height and width')
    args = parser.parse_args()

    preprocess_folder(args)


main()
