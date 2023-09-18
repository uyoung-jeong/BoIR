import argparse
import os
import pprint
from multiprocessing import Process, Queue
from collections import OrderedDict
from turtle import pos

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
from tqdm import tqdm
import numpy as np
import time

import _init_paths
import models
from config import get_cfg, update_config
from core.evaluator import Evaluator
from core.trainer import AverageMeter
from dataset import make_test_dataloader
from utils.logging import create_checkpoint, setup_logger
from utils.transforms import get_multi_scale_size, resize_align_multi_scale, get_final_preds
from utils.nms import oks_nms

from utils.utils import _print_name_value

def parse_args():
    parser = argparse.ArgumentParser(description='Test BoIR')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--gpus',
                        help='gpu ids for eval',
                        default='0',
                        type=str)
    args = parser.parse_args()
    return args

def worker(gpu_id, dataset, indices, cfg, args, logger, final_output_dir, pred_queue, gpu_list):
    if cfg.DDP:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[gpu_id]

    model = models.create(cfg.MODEL.NAME, cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        try:
            #model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
            model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        except RuntimeError as e:
            state_dict = torch.load(cfg.TEST.MODEL_FILE)['best_state_dict']
            #model.load_state_dict(state_dict, strict=True)
            model.load_state_dict(state_dict, strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, "model_best.pth.tar")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:.3e}")

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    sub_dataset = torch.utils.data.Subset(dataset, indices) if cfg.DDP else dataset
    data_loader = torch.utils.data.DataLoader(
        sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )])
    infer_time_meter = AverageMeter()
    all_preds = []
    data_loader = tqdm(data_loader, dynamic_ncols=True)
    for i, batch_inputs in enumerate(data_loader):
        img_id = batch_inputs['image_id'].item()
        image = batch_inputs['image'][0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        with torch.no_grad():
            image_resized, center, scale = resize_align_multi_scale(image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0)
            image_resized = transforms(image_resized).unsqueeze(0) # [1, 3, 512, w]

            inputs = [{'image': image_resized}]
            if cfg.TEST.FLIP_TEST:
                image = torch.flip(image_resized, [3])
                inputs.append({'image': image})

            start_time = time.time()
            instances = model(inputs)
            
            if 'poses' not in instances: continue

            poses = instances['poses'].cpu().numpy()
            scores = instances['scores'].cpu().numpy()

            poses = get_final_preds(poses, center, scale, [base_size[0], base_size[1]])
            # perform nms
            keep, _ = oks_nms(poses, scores, cfg.TEST.OKS_SCORE, np.array(cfg.TEST.OKS_SIGMAS) / 10.0)
            
            end_time = time.time()
            infer_time_meter.update(end_time-start_time)

            for _keep in keep:
                all_preds.append({
                    "keypoints": poses[_keep][:, :3].reshape(-1, ).astype(float).tolist(),
                    "image_id": img_id,
                    "score": float(scores[_keep]),
                    "category_id": 1
                })
            
    # close tqdm
    data_loader.close()
    pred_queue.put_nowait(all_preds)
    print(f"Average model inference time: {infer_time_meter.avg}")


def main():
    args = parse_args()
    cfg = get_cfg()
    update_config(cfg, args)

    final_output_dir = create_checkpoint(cfg, 'valid')
    logger, _ = setup_logger(final_output_dir, 0, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    print(f'final_output_dir: {final_output_dir}')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.multiprocessing.set_start_method('spawn') # without this, raises "RuntimeError: Cannot re-initialize CUDA in forked subprocess."

    test_dataset, _ = make_test_dataloader(cfg)

    total_size = len(test_dataset)
    pred_queue = Queue(100)
    workers = []
    gpu_list = args.gpus.split(',')
    num_gpus = len(gpu_list)

    if not cfg.DDP:
        indices = list(range(0, total_size, num_gpus))
        worker(args.gpus, test_dataset, indices, cfg, args, logger, final_output_dir, pred_queue, gpu_list)
    else:
        for i in range(num_gpus):
            indices = list(range(i, total_size, num_gpus))
            p = Process(
                target=worker,
                args=(
                    i, test_dataset, indices, cfg, args, logger, final_output_dir, pred_queue, gpu_list
                )
            )
            p.start()
            workers.append(p)
            logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(indices)))
    
    all_preds = []
    for idx in range(num_gpus):
        all_preds += pred_queue.get()
    if cfg.DDP:
        for p in workers:
            p.join()

    evaluator = Evaluator(cfg, final_output_dir)
    info_str = evaluator.evaluate(all_preds)
    name_values = OrderedDict(info_str)

    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == "__main__":
    main()
