import argparse
import os
import copy
import logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.cuda import amp
from torch.utils.data.dataloader import DataLoader
from model import EBSD, SRResNet_RGBY
from dataset import Dataset
from utils import AverageMeter, ProgressMeter, calc_psnr, preprocess, DataParallelCriterion
from PIL import Image
# from parallel import DataParallelCriterion
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.data_parallel import DataParallel


def main():
    args = parser.parse_args()

    """ GPU device 설정 """
    gpu_devices = ",".join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    """ 사용가능한 GPU 개수 반환 """
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(gpu))

    # scale별 weight 저장 경로 설정
    args.outputs_dir = os.path.join(args.outputs_dir, "x{}".format(args.scale))

    # 저장 경로 없을 시 생성
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    args.rank = args.rank * ngpus_per_node + gpu

    """ 각 GPU마다 분산 학습을 위한 초기화 실행 """
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    """ 모델 생성 """
    print("==> 모델 생성중..")
    torch.cuda.set_device(args.gpu)
    model = EBSD().cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("모델의 파라미터 수 : ", num_params)

    criterion = nn.L1Loss()
    criterion = DataParallelCriterion(criterion, device_ids=[args.gpu])
    optimizer = optim.Adam(model.parameters(), args.lr, (0.9, 0.999))

    # 체크포인트 weight 불러오기
    if os.path.exists(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        best_psnr = checkpoint["best_psnr"]

    # 스케줄러 설정
    psnr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = amp.GradScaler()

    train_dataset = Dataset(args.train_file, args.patch_size, args.scale)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    eval_dataset = Dataset(args.eval_file, args.patch_size, args.scale)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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
            prefix=f"Epoch: [{epoch}]",
        )

        # 트레이닝 Epoch 시작
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.cuda(args.gpu)
            hr = hr.cuda(args.gpu)

            optimizer.zero_grad()

            with amp.autocast():
                preds = model(lr)
                loss = criterion(preds, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), len(lr))
            psnr.update(calc_psnr(preds, hr), len(lr))

            if i % 100 == 0:
                progress.display(i)

        psnr_scheduler.step()

        # epoch 별 가중치를 설정한 경로에 저장
        torch.save(
            model.state_dict(),
            os.path.join(args.outputs_dir, "epoch_{}.pth".format(epoch)),
        )

        model.eval()

        # 테스트 Epoch 시작
        model.eval()
        with torch.no_grad():
            for i, (lr, hr) in enumerate(eval_dataloader):
                lr = lr.cuda(args.gpu)
                hr = hr.cuda(args.gpu)
                preds = model(lr)

        if psnr.avg > best_psnr:
            best_psnr = psnr.avg
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, "best.pth"))

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "best_psnr": best_psnr,
            },
            os.path.join(args.outputs_dir, "epoch_{}.pth".format(epoch)),
        )

        # """ 에스파 이미지 테스트 """
        # with torch.no_grad():
        #     lr = test_image.to(device)
        #     preds = model(lr)
        #     vutils.save_image(preds.detach(), os.path.join(args.outputs_dir, f"PSNR_{epoch}.png"))


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    # Argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    parser.add_argument('--gpu_devices', type=int, nargs='+', required=True)
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:4179', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument("--weights_file", type=str)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint-file", type=str, default="checkpoint-file.pth")

    main()
