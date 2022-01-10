import argparse
import os
import sys

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import collate, scatter
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model  # isort:skip  # noqa
from mmgen.datasets.pipelines import Compose
from mmgen.models import BaseTranslationModel
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Translation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image_path', help='Image file path')
    parser.add_argument(
        '--target-domain', type=str, default=None, help='Desired image domain')
    parser.add_argument(
        '--save-path', type=str, default=None,
        help='path to save translation sample')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    args = parser.parse_args()
    return args


def sample_img2img_model(model, image_path, target_domain=None, **kwargs):
    """Sampling from translation models.

    Args:
        model (nn.Module): The loaded model.
        image_path (str): File path of input image.
        style (str): Target style of output image.
    Returns:
        Tensor: Translated image tensor.
    """
    assert isinstance(model, BaseTranslationModel)

    # get source domain and target domain
    if target_domain is None:
        target_domain = model._default_domain
    source_domain = model.get_other_domains(target_domain)[0]

    cfg = model._cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    inference_pipeline = Compose(cfg.inference_pipeline)

    # prepare data
    data = dict()
    # dirty code to deal with inference data pipeline
    data['pair_path'] = image_path
    data[f'img_{source_domain}_path'] = image_path

    data = inference_pipeline(data)
    if device.type == 'cpu':
        data = collate([data], samples_per_gpu=1)
        data['meta'] = []
    else:
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    source_image = data[f'img_{source_domain}']
    # forward the model
    with torch.no_grad():
        results = model(
            source_image,
            test_mode=True,
            target_domain=target_domain,
            **kwargs)
    output = results['target']
    return output


def get_full_save_path(save_path, image_path, ckpt_path):
    if save_path is None:
        save_path = os.path.join('work_dirs', 'demos')

    fn = os.path.splitext(os.path.split(os.path.normpath(image_path))[1])[0]

    abs_path, _ = os.path.split(ckpt_path)
    _, ckpt_folder_name = os.path.split(abs_path)
    segments = ckpt_folder_name.split('_')
    if 'serial' in segments or 'pidinet' in segments or 'th' == segments[-1]:
        method_name = '{}_{}'.format(segments[-2], segments[-1])
    else:
        method_name = segments[-1]

    new_fn = f'{fn}_{method_name}.jpg'
    full_save_path = os.path.join(save_path, new_fn)
    return full_save_path


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    results = sample_img2img_model(model, args.image_path, args.target_domain)
    results = (results[:, [2, 1, 0]] + 1.) / 2.

    # save images
    args.save_path = get_full_save_path(
        args.save_path, args.image_path, args.checkpoint)
    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(results, args.save_path)


if __name__ == '__main__':
    main()
