import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from model import EBSD
from utils import preprocess, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--num-blocks', type=str, default=8)
    parser.add_argument('--num-layers', type=str, default=8)
    parser.add_argument('--num-growth-rate', type=str, default=16)
    parser.add_argument('--num-channels', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = EBSD(scale_factor=args.scale, num_channels=args.num_channels, growth_rate=args.num_growth_rate, n_blocks=args.num_blocks, n_layers=args.num_layers).to(device)
    state_dict = model.state_dict()
    try:
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
    except:
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage)['model_state_dict'].items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr = preprocess(lr).to(device)
    hr = preprocess(hr).to(device)
    bic = preprocess(bicubic).to(device)

    with torch.no_grad():
        preds = model(lr)

    sr_psnr = calc_psnr(hr, preds)
    bic_psnr = calc_psnr(hr, bic)
    
    print('SR PSNR: {:.2f}'.format(sr_psnr))
    print('BIC PSNR: {:.2f}'.format(bic_psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0)

    output = np.array(preds).transpose([1,2,0])
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_EBSD_x{}.'.format(args.scale)))
