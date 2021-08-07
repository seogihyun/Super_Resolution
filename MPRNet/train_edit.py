
import argparse
import os
import copy
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import time
import numpy as np
import utils
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.data_parallel import DataParallel
from warmup_scheduler import GradualWarmupScheduler
from utils import AverageMeter, ProgressMeter, calc_psnr, preprocess, DataParallelCriterion
from config import Config 
from MPRNet import MPRNet
from tqdm import tqdm
from pdb import set_trace as stx
from PIL import Image
from data_RGB import get_training_data, get_validation_data
from losses import CharbonnierLoss, EdgeLoss

def main():
    args = parser.parse_args()

    """ GPU device 설정 """
    gpu_devices = ",".join([str(id) for id in args.gpu_devices])

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
    torch.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.rank = args.rank * ngpus_per_node + gpu

    start_epoch = 1
    mode = args.mode
    session = args.model

    result_dir = os.path.join(args.outputs_dir, mode, 'results', session)
    model_dir  = os.path.join(args.outputs_dir, mode, 'models',  session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = args.train_file
    val_dir = args.eval_file

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
    model = MPRNet().cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("모델의 파라미터 수 : ", num_params)

    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    criterion_char = DataParallelCriterion(criterion_char, device_ids=[args.gpu])
    criterion_edge = DataParallelCriterion(criterion_edge, device_ids=[args.gpu])
    optimizer = optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-8)

    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs-warmup_epochs, eta_min=args.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # 체크포인트 weight 불러오기
    if os.path.exists(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        best_psnr = checkpoint["best_psnr"]
   
    """ Dataset load """
    train_dataset = get_training_data(args.train_file, {'patch_size':args.patch_size})
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True, sampler=train_sampler)

    eval_dataset = get_validation_data(args.eval_file, {'patch_size':args.patch_size})
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=128, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, args.num_epochs+1))
    print('===> Loading datasets')

    best_weights = copy.deepcopy(model.state_dict())
    best_psnr = 0
    best_epoch = 0

    for epoch in range(start_epoch, args.num_epochs+1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model.train()

        # 트레이닝 Epoch 시작
        for i, data in enumerate(tqdm(train_loader),0):

            # zero_grad
            for param in model.parameters():
                param.grad = None

            target = data_val[0].cuda(args.gpu)
            input_ = data_val[1].cuda(args.gpu)

            restored = model(input_)
    
            # Compute loss at each stage
            loss_char = np.sum([criterion_char(restored[j], target) for j in range(len(restored))])
            loss_edge = np.sum([criterion_edge(restored[j], target) for j in range(len(restored))])
            loss = (loss_char) + (0.05*loss_edge)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch%args.eval_after_every== 0:
                model.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((eval_loader), 0):
                    target = data_val[0].cuda(args.gpu)
                    input_ = data_val[1].cuda(args.gpu)

                    with torch.no_grad():
                        restored = model(input_)
                    restored = restored[0]

                    for res, tar in zip(restored, target):
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))

                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    torch.save({'epoch': epoch, 
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir, f"{model}_best.pth"))

                print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
                 
                # epoch 별 가중치를 설정한 경로에 저장
                torch.save({'epoch': epoch, 
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,f"{model}_{args.scale}_epoch_{epoch}.pth")) 
        

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"{model}_{args.scale}_latest.pth")) 

        
        
       
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
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint-file", type=str, default="checkpoint-file.pth")
    parser.add_argument("--mode", type=str, default="All_of_Denoising_Deraining_Deblurring")
    parser.add_argument("--model", type=str, default="MPRNet")
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--eval_after_every", type=int, default=10)

    main()
