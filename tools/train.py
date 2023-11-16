import argparse
import os
import pprint
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import _init_paths
import models
from config import get_cfg, update_config
from core.trainer import Trainer
from dataset import make_train_dataloader
from utils.logging import create_checkpoint, setup_logger
from utils.utils import get_optimizer, save_checkpoint

import torchvision
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from utils.utils import cosine_scheduler, NativeScaler

import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train BoIR')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--output_dir',
                        help='output directory to continue',
                        type=str, default='')

    # distributed training
    parser.add_argument('--gpus',
                        help='gpu ids for ddp training',
                        type=str)
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--port',
                        default='23459',
                        type=str,
                        help='port used to set up distributed training')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = get_cfg()
    update_config(cfg, args)

    final_output_dir = create_checkpoint(cfg, 'train', output_dir=args.output_dir)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    dist_url = args.dist_url + ':{}'.format(args.port)

    ngpus_per_node = torch.cuda.device_count()
    if cfg.DDP:
        world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(world_size, dist_url, final_output_dir, args))
    else:
        main_worker(0, 1, dist_url, final_output_dir, args)

def main_worker(rank, world_size, dist_url, final_output_dir, args):
    cfg = get_cfg()
    update_config(cfg, args)
    # setup logger
    logger, _ = setup_logger(final_output_dir, rank, 'train')
    if not cfg.DDP or (cfg.DDP and rank == 0):
        logger.info(pprint.pformat(args))
        logger.info(cfg)
        logger.info(f"final_output_dir: {final_output_dir}")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DDP:
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.format(dist_url, world_size, rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )

    model = models.create(cfg.MODEL.NAME, cfg, is_train=True)

    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tblog')),
        'train_global_steps': 0
    }

    if cfg.DDP:
        if cfg.MODEL.SYNC_BN:
            print('use sync bn')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    train_loader = make_train_dataloader(cfg, distributed=cfg.DDP)
    if not cfg.DDP or (cfg.DDP and rank == 0):
        logger.info(train_loader.dataset)

    best_perf = -1
    best_epoch = -1
    last_epoch = -1
    optimizer = get_optimizer(cfg, model.parameters())
    scaler = NativeScaler() if cfg.TRAIN.AMP else None

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'model', 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint.keys():
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    # save config file
    if not cfg.DDP or (cfg.DDP and rank == 0):
        src_folder = os.path.join(final_output_dir, 'src')
        if not os.path.exists(os.path.join(src_folder, 'lib')):
            shutil.copytree('lib', os.path.join(src_folder, 'lib'))
            shutil.copytree('tools', os.path.join(src_folder, 'tools'))
            shutil.copy2(args.cfg, src_folder)
        else:
            logger.info("=> src files are already existed in: {}".format(src_folder))

    niter_per_epoch = len(train_loader)
    lr_schedule_values = None
    if cfg.TRAIN.LR_SCHEDULER == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
                last_epoch=last_epoch
            )
    elif cfg.TRAIN.LR_SCHEDULER == 'cosine':
        lr_schedule_values = cosine_scheduler(cfg.TRAIN.LR, cfg.TRAIN.MIN_LR, 
                                            epochs=cfg.TRAIN.END_EPOCH, niter_per_ep=niter_per_epoch, 
                                            warmup_epochs=cfg.TRAIN.WARMUP_EPOCHS, start_warmup_value=cfg.TRAIN.WARMUP_START_VALUE)

    
    trainer = Trainer(cfg, model, rank, final_output_dir, writer_dict, scaler, niter_per_epoch, lr_schedule_values)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if cfg.DDP:
            dist.barrier()
            train_loader.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader, optimizer)

        if cfg.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()
        
        if cfg.DDP:
            dist.barrier()
        
        # save checkpoint
        if (not cfg.DDP or (cfg.DDP and rank == 0)) and ((epoch % 2 == 0) or (epoch+1==cfg.TRAIN.END_EPOCH)):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': -1,
                'optimizer': optimizer.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict()
            }, False, final_output_dir)

    if not cfg.DDP or (cfg.DDP and rank == 0):
        time.sleep(60) # make sure everything is written to tensorboard log file
        writer_dict['writer'].flush()
        final_model_state_file = os.path.join(
            final_output_dir, 'model', 'final_state{}.pth.tar'.format(rank)
        )
        logger.info('saving final model state to {}'.format(final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()

    if cfg.DDP:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
