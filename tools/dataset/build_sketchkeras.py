import os
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn

from models_sketchkeras import SketchKeras


def get_args():
    dic = {'ckpt': os.path.join('ckpt', 'model.pth')}
    args = SimpleNamespace(**dic)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'cuda' in args.device:
        cudnn.benchmark = True

    return args


def get_sketchkeras():
    args = get_args()

    model = SketchKeras().to(args.device)

    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    print(f'Loaded checkpoint from: {args.ckpt}.')

    model.eval()
    return model


if __name__ == '__main__':
    model = get_sketchkeras()
