import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN_x
from utils import preprocess, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--image_file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN_x(scale_factor=args.scale).to(device)
    
    state_dict = model.state_dict()
    
    # (map_location=lambda storage, loc: storage) : a way of load model on CPU
    # GPU에서 학습한 weights를 CPU에서 적용
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys(): # n : index, p : parameters
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)

    # bicubic upscaling
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    
    # 추론한 결과물 저장할 파일명 변경
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    # ycbcr값
    lr = preprocess(lr, device)
    hr = preprocess(hr, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)
        print('pred.shape : ', preds.shape)

    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0)
    print('pred shape: ', preds.shape)

    # output : (c,h,w) -> (h,w,c)로 변경
    output = np.array(preds).transpose([1, 2, 0])
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    print('output shape : ', output.shape)
    # pil image 형태로 변환 (pil file : PNG, JPEG, GIF, BMP 등을 가지고 있는 파일 포맷 라이브러리)
    output = pil_image.fromarray(output)

    # 저장할 파일명 변경
    output.save(args.image_file.replace('.', '_fsrcnn-s_x{}.'.format(args.scale)))
