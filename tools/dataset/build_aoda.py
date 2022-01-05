from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn

from models_aoda import create_aoda


def get_args():
    dic = {}
    args = SimpleNamespace(**dic)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'cuda' in args.device:
        cudnn.benchmark = True

    return args


def get_aoda():
    args = get_args()

    model = create_aoda().to(args.device)
    model.eval()
    return model


if __name__ == '__main__':
    model = get_aoda()
