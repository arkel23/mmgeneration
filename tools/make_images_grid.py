import os
import argparse

import einops
import numpy as np
from PIL import Image
from skimage.transform import resize


ALGOS_TRAD = ['linear', 'canny', 'xdog', 'xdog_th', 'serial0.3',
              'serial0.4', 'serial0.5']
ALGOS_DL = ['sketchkeras', 'aoda',
            'pidinet_-1', 'pidinet_0', 'pidinet_1', 'pidinet_2', 'pidinet_3']
COMBINATIONS = ['all', 'dl', 'trad']
ALGOS = ALGOS_TRAD + ALGOS_DL + COMBINATIONS


def read_image(fp):
    img = Image.open(fp).convert('RGB')
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def gallery(array, ncols=8):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    result = einops.rearrange(
        array, '(b1 b2) h w c -> (b1 h) (b2 w) c', b2=ncols)
    return result


def make_grid(args):

    fn = os.path.split(os.path.normpath(args.path_images))[1]

    image_list = []
    for mode in args.preprocess:
        fp = f'{args.path_images}_{mode}.jpg'
        assert os.path.isfile(fp), f'{fp} file does not exist.'

        img = read_image(fp)
        img = img.resize((args.res_hw, args.res_hw))
        image_list.append(img)

    img_row = np.stack([np.array(img) for img in image_list], axis=0)
    img_grid = gallery(img_row, ncols=args.ncols)
    img_grid = Image.fromarray(img_grid, 'RGB')

    os.makedirs(args.path_save, exist_ok=True)
    new_fn = os.path.join(args.path_save, f'{fn}_grid.png')
    img_grid.save(new_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_images', type=str, required=True,
                        help='base folder and fn for images')
    parser.add_argument('--path_save', type=str, required=True,
                        help='folder to save new image grid')
    parser.add_argument('--preprocess', nargs='+', type=str, choices=ALGOS,
                        default='linear', help='greyscale/sketch conversion')
    parser.add_argument('--res_hw', type=int, default=128,
                        help='resized height and width')
    parser.add_argument('--ncols', type=int, default=8, help='images per row')
    args = parser.parse_args()
    make_grid(args)


main()
