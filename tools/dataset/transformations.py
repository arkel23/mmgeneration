import os
import os.path as osp
import argparse

import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian
import torch
import torchvision.transforms as transforms


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


def convert_sketchkeras(img, model, thresh=0.1, image_size=512):
    '''
    https://github.com/lllyasviel/sketchKeras
    https://github.com/higumax/sketchKeras-pytorch
    '''
    device = next(model.parameters()).device

    width, height = img.size
    tsfm_resize_og = transforms.Resize((height, width),
                                       transforms.InterpolationMode.BICUBIC)

    if width > height:
        new_width, new_height = (image_size, int(image_size / width * height))
    else:
        new_width, new_height = (int(image_size / height * width), image_size)

    img_array = np.array(img.convert('RGB'))
    img_resized = cv2.resize(img_array, (new_width, new_height))
    new_height, new_width, c = img_resized.shape

    blurred = cv2.GaussianBlur(img_resized, (0, 0), 3)
    highpass = img_resized.astype(int) - blurred.astype(int)
    highpass = highpass.astype(float) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((image_size, image_size, 3), dtype=float)
    ret[0:new_height, 0:new_width, 0:c] = highpass

    x = ret.reshape(1, *ret.shape).transpose(3, 0, 1, 2)
    x = torch.tensor(x).float().to(device)

    with torch.no_grad():
        pred = model(x).squeeze()

    pred = pred.cpu().detach().numpy()

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)

    pred = pred[:new_height, :new_width]
    pred = Image.fromarray(pred, 'L')
    output_og_size = tsfm_resize_og(pred)
    return output_og_size


def convert_aoda(img, model, image_size=512):
    '''
    Adversarial Open Domain Adaptation for Sketch-to-Photo Synthesis
    https://github.com/Mukosame/Anime2Sketch
    https://github.com/Mukosame/AODA
    '''
    device = next(model.parameters()).device

    tsfm = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    width, height = img.size
    tsfm_resize_og = transforms.Resize((height, width),
                                       transforms.InterpolationMode.BICUBIC)

    img_tensor = tsfm(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)

    # tensor to img
    out = out.data[0].cpu().float().numpy()
    out = np.tile(out, (3, 1, 1))
    out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0

    out = Image.fromarray(out.astype(np.uint8), 'RGB')
    out_og_size = tsfm_resize_og(out)
    return out_og_size.convert('L')


def convert_pidinet(img, model, alg_name):
    '''
    Pixel Difference Convolutional Networks
    https://github.com/zhuoinoulu/pidinet
    https://arxiv.org/abs/2108.07009
    '''
    device = next(model.parameters()).device
    filt = int(alg_name.split('_')[1])

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    img = tsfm(img.convert('RGB')).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        results = model(img)

    result = torch.squeeze(results[filt]).cpu().numpy()
    result = cv2.normalize(src=result, dst=None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    inv = cv2.bitwise_not(result)
    return Image.fromarray(inv.astype(np.uint8), 'L')


def convert_image(args, img, model=None):
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
    elif args.preprocess == 'sketchkeras':
        img_new = convert_sketchkeras(img, model)
    elif args.preprocess == 'aoda':
        img_new = convert_aoda(img, model)
    elif 'pidinet' in args.preprocess:
        img_new = convert_pidinet(img, model, args.preprocess)
    else:
        raise NotImplementedError
    return img_new


if __name__ == '__main__':
    algos = ['linear', 'canny', 'xdog', 'xdog_th', 'xdog_serial',
             'sketchkeras', 'aoda',
             'pidinet_-1', 'pidinet_0', 'pidinet_1', 'pidinet_2', 'pidinet_3']
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
        elif alg == 'sketchkeras':
            from build_sketchkeras import get_sketchkeras
            model = get_sketchkeras()
            img_new = convert_image(args, img, model)
            img_new.save(osp.join(args.save, f'{fn}_{alg}.jpg'))
        elif alg == 'aoda':
            from build_aoda import get_aoda
            model = get_aoda()
            img_new = convert_image(args, img, model)
            img_new.save(osp.join(args.save, f'{fn}_{alg}.jpg'))
        elif 'pidinet' in alg:
            from build_pidinet import get_pidinet
            model = get_pidinet()
            img_new = convert_image(args, img, model)
            img_new.save(osp.join(args.save, f'{fn}_{alg}.jpg'))
        else:
            img_new = convert_image(args, img)
            img_new.save(osp.join(args.save, f'{fn}_{alg}.jpg'))
