import os
import os.path as osp
import argparse

import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian


def convert_linear(img):
    return img.convert('L')


def convert_canny(img, th1=100, th2=200):
    # https://learnopencv.com/edge-detection-using-opencv/
    img_blur = cv2.GaussianBlur(np.array(img.convert('L')), (3, 3), 0)
    edges = cv2.Canny(img_blur, th1, th2)
    inv = cv2.bitwise_not(edges)
    return Image.fromarray(inv.astype('uint8'), 'L')


def convert_xdog(img, sigma=0.8, k=1.6, gamma=0.98, eps=-0.1, phi=200,
                 thresh=False):
    '''
    https://github.com/aaroswings/XDoG-Python/blob/main/XDoG.py
    sigma=0.8, k=1.6, gamma=0.98, eps=-0.1, phi=200

    https://github.com/heitorrapela/xdog/blob/master/main.py
    sigma=0.5, k=1.6, gamma=1, eps=1, phi=1
    these values do not work and lead to all black results (designed for uint8)

    https://subscription.packtpub.com/book/data/9781789537147/1/ch01lvl1sec06/creating-pencil-sketches-from-images
    sigma=0.5, k=200, gamma=1, eps=0.01, phi=10
    these values do get edges but does not look like a sketch or manga
    '''
    img = np.array(img.convert('L'))

    g_filtered_1 = gaussian(img, sigma)
    g_filtered_2 = gaussian(img, sigma * k)

    z = g_filtered_1 - gamma * g_filtered_2

    z[z < eps] = 1.

    mask = z >= eps
    z[mask] = 1. + np.tanh(phi * z[mask])

    if thresh:
        mean = z.mean()
        z[z < mean] = 0.
        z[z >= mean] = 1.

    z = cv2.normalize(src=z, dst=None, alpha=0, beta=255,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(z.astype('uint8'), 'L')


def convert_xdog_serial(img, sigma=0.5, k=4.5, gamma=19, eps=0.01, phi=10**9):
    '''
    https://github.com/SerialLain3170/Colorization/blob/c920440413429af588e0b6bd6799640d1feda68e/nohint_pix2pix/xdog.py
    sigma_range=[0.3, 0.4, 0.5], k_sigma=4.5, p=19, eps=0.01, phi=10**9,
    sigma_large = sigma * k_sigma
    p is similar to gamma but also multiplies by first gaussian
    '''
    img = np.array(img.convert('L'))

    g_filtered_1 = gaussian(img, sigma)
    g_filtered_2 = gaussian(img, sigma * k)

    z = (1+gamma) * g_filtered_1 - gamma * g_filtered_2

    si = np.multiply(img, z)

    edges = np.zeros(si.shape)
    si_bright = si >= eps
    si_dark = si < eps
    edges[si_bright] = 1.0
    edges[si_dark] = 1.0 + np.tanh(phi * (si[si_dark] - eps))

    edges = cv2.normalize(src=edges, dst=None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(edges.astype('uint8'), 'L')


def convert_image(args, img):
    if args.preprocess == 'linear':
        img_new = convert_linear(img)
    elif args.preprocess == 'canny':
        img_new = convert_canny(img)
    elif args.preprocess == 'xdog':
        img_new = convert_xdog(img)
    elif args.preprocess == 'xdog_th':
        img_new = convert_xdog(img, thresh=True)
    elif args.preprocess == 'xdog_serial':
        img_new = convert_xdog_serial(img, sigma=args.sigma)
    else:
        raise NotImplementedError
    return img_new


if __name__ == '__main__':
    algos = ['linear', 'canny', 'xdog', 'xdog_th', 'xdog_serial']
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        default=osp.join('samples', 'test.jpg'),
                        help='test img path')
    parser.add_argument('--save', type=str, default='results',
                        help='save folder')
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    img = Image.open(args.test_path)
    fn = osp.splitext(osp.split(osp.normpath(args.test_path))[1])[0]

    sigmas = [0.3, 0.4, 0.5]

    for alg in algos:
        args.preprocess = alg
        if alg == 'xdog_serial':
            for sigma in sigmas:
                args.sigma = sigma
                img_new = convert_image(args, img)
                img_new.save(osp.join(args.save, f'{fn}_{alg}_{sigma}.jpg'))
        else:
            img_new = convert_image(args, img)
            img_new.save(osp.join(args.save, f'{fn}_{alg}.jpg'))
