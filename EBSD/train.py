import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'
import copy
import logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch import nn
from torch.cuda import amp
from torch.utils.data.dataloader import DataLoader
from model import EBSD
from dataset import Dataset
from utils import AverageMeter, ProgressMeter, calc_psnr, preprocess
from PIL import Image


# # 에스파 테스트 이미지 경로 설정
# test_image_path = 'examples/aespa.png'
# # 에스파 테스트 이미지 불러오기
# test_image = Image.open(test_image_path).convert('RGB')
# # 에스파 테스트 이미지 전처리
# test_image = preprocess(test_image)



if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    # Argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--weights_file', type=str)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_epochs', type=int, default=50000)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint-file', type=str, default='checkpoint-file.pth')
    args = parser.parse_args()


    # scale별 weight 저장 경로 설정
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    # 저장 경로 없을 시 생성
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    model = EBSD().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), args.lr, (0.9, 0.999))


    # 체크포인트 weight 불러오기
    if os.path.exists(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    # 스케줄러 설정 
    psnr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = amp.GradScaler()

    train_dataset = Dataset(args.train_file, args.patch_size, args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = Dataset(args.eval_file, args.patch_size, args.scale)
    eval_dataloader = DataLoader(
                                dataset=eval_dataset, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True
                                )

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        progress = ProgressMeter(
            num_batches=len(train_dataloader),
            meters=[losses, psnr],
            prefix=f"Epoch: [{epoch}]"
        )

        # 트레이닝 Epoch 시작
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()

            with amp.autocast():
                preds = model(lr)
                loss = criterion(preds, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), len(lr))
            psnr.update(calc_psnr(preds, hr), len(lr))

            if i%100==0:
                progress.display(i)

        psnr_scheduler.step()
        
        # epoch 별 가중치를 설정한 경로에 저장
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()


        # 테스트 Epoch 시작
        model.eval()
        with torch.no_grad():
            for i, (lr, hr) in enumerate(eval_dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                preds = model(lr)
                

        if psnr.avg > best_psnr:
            best_psnr = psnr.avg
            torch.save(
                model.state_dict(), os.path.join(args.outputs_dir, 'best.pth')
            )

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_psnr': best_psnr,
            }, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch))
        )

        # """ 에스파 이미지 테스트 """
        # with torch.no_grad():
        #     lr = test_image.to(device)
        #     preds = model(lr)
        #     vutils.save_image(preds.detach(), os.path.join(args.outputs_dir, f"PSNR_{epoch}.png"))
