import os
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn

import models_pidinet
from models_pidinet.convert_pidinet import convert_pidinet


def get_args():
    dic = {'model': 'pidinet_converted', 'sa': True, 'dil': True,
           'config': 'carv4', 'gpu': '0',
           'ckpt': os.path.join('ckpt', 'table7_pidinet.pth')}
    args = SimpleNamespace(**dic)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.use_cuda = torch.cuda.is_available()
    if args.use_cuda:
        cudnn.benchmark = True

    return args


def load_checkpoint(args):
    print("=> loading checkpoint from '{}'".format(args.ckpt))
    return torch.load(args.ckpt, map_location='cpu')


def get_pidinet():
    '''
    Pixel Difference Convolutional Networks
    https://github.com/zhuoinoulu/pidinet
    https://arxiv.org/abs/2108.07009
    '''
    args = get_args()

    # Create model
    model = getattr(models_pidinet, args.model)(args)

    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    checkpoint = load_checkpoint(args)
    model.load_state_dict(
        convert_pidinet(checkpoint['state_dict'], args.config))

    model.eval()
    return model


def simple_inference(model):
    from PIL import Image
    import numpy as np
    import torchvision
    import torchvision.transforms as transforms

    img = Image.open(os.path.join('samples', 'test.jpg')).convert('RGB')

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    img = tsfm(img).unsqueeze(0)
    img = img.cuda() if torch.cuda.is_available() else img
    _, _, H, W = img.shape

    with torch.no_grad():
        results = model(img)
        result = torch.squeeze(results[-1]).cpu().numpy()

    results_all = torch.zeros((len(results), 1, H, W))
    for i in range(len(results)):
        results_all[i, 0, :, :] = results[i]

    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save('edges.jpg')
    torchvision.utils.save_image(1-results_all, 'edges_all.jpg')


if __name__ == '__main__':
    model = get_pidinet()
    simple_inference(model)
