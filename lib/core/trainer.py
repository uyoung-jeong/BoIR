import logging
import time

from tqdm import tqdm
import torch
import math

class Trainer(object):
    def __init__(self, cfg, model, rank, output_dir, writer_dict, scaler, niter_per_epoch, lr_schedule_values=None):
        self.model = model
        self.output_dir = output_dir
        self.rank = rank
        self.print_freq = cfg.PRINT_FREQ
        self.writer_dict = writer_dict
        self.scaler = scaler
        self.use_amp = cfg.TRAIN.AMP
        self.clip_grad = cfg.TRAIN.CLIP_GRAD
        self.niter_per_epoch = niter_per_epoch
        self.lr_schedule_values = lr_schedule_values

        self.device_ids = self.model.device_ids
        self.model_name = cfg.MODEL.NAME

    def train(self, epoch, data_loader, optimizer):
        logger = logging.getLogger("Training")
        lr_schedule_values = None if self.lr_schedule_values is None else self.lr_schedule_values[epoch*self.niter_per_epoch:(epoch+1)*self.niter_per_epoch]

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meters = dict()
        grad_norm_meter = AverageMeter()
        loss_scale_meter = AverageMeter()

        aux_meters = dict()

        self.model.train()

        end = time.time()
        pbar = tqdm(total=len(data_loader), dynamic_ncols=True) if self.rank == 0 else None
        for i, batched_inputs in enumerate(data_loader):
            if i > 2:
                break
            if lr_schedule_values is not None: # cosine lr scheduler
                lr_value = lr_schedule_values[i]
                for pi, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = lr_value

            num_images = batched_inputs['image'].shape[0] if isinstance(batched_inputs, dict) else len(batched_inputs)
            data_time.update(time.time() - end)

            loss_dict, aux = self.model(batched_inputs)

            loss = 0.0
            for k in loss_dict.keys():
                loss_dict[k] = loss_dict[k].sum() / num_images

                # https://github.com/facebookresearch/ConvNeXt/blob/main/engine.py
                if not math.isfinite(loss_dict[k]): # this could trigger when AMP is used
                    print(f"Loss is {loss_dict[k]} at {k}, stopping training")
                    assert math.isfinite(loss_dict[k])

                if k not in loss_meters.keys():
                    loss_meters[k] = AverageMeter()
                loss_meters[k].update(loss_dict[k].item(), num_images)
                loss = loss + loss_dict[k]

            if self.use_amp:
                optimizer.zero_grad()
                grad_norm = self.scaler(loss, optimizer, clip_grad=self.clip_grad,
                                parameters=self.model.parameters(), create_graph=False,
                                update_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                grad_norm = None

            batch_time.update(time.time() - end)
            end = time.time()

            if aux is not None and self.rank == 0:
                for k,v in aux.items():
                    if k not in aux_meters.keys():
                        aux_meters[k] = AverageMeter()
                    aux_meters[k].update(v.mean().item(), 1)
            
            torch.cuda.synchronize()

            if grad_norm is not None:
                if not torch.isnan(grad_norm).any() and not torch.isinf(grad_norm).any():
                    grad_norm_meter.update(grad_norm)
                elif self.rank == 0:
                    print(f"nan or inf gradient detected at {i}th step: {grad_norm}")
                loss_scale_value = self.scaler.state_dict()["scale"]
                loss_scale_meter.update(loss_scale_value)

            if i % self.print_freq == 0 and self.rank == 0:
                msg = f"[{epoch}][{i}/{len(data_loader)}]"
                for k,v in loss_meters.items():
                    msg += f", {_get_loss_info(v, k)}"
                msg += f", lr: {optimizer.param_groups[0]['lr']:.3e}, grad norm: {grad_norm_meter.avg:.3e}"
                #logger.info(msg)
                if pbar is not None:
                    pbar.set_description(msg)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        # write loss in tensorboard
        if self.rank == 0:
            writer = self.writer_dict['writer']
            for k,v in loss_meters.items():
                writer.add_scalar(k, v.avg, epoch)
            writer.add_scalar('grad norm', grad_norm_meter.avg, epoch)
            writer.add_scalar('loss scale', loss_scale_meter.avg, epoch)
            for k,v in aux_meters.items():
                writer.add_scalar(k, v.avg, epoch)
            writer.add_scalar('batch_time', batch_time.avg, epoch)
            writer.add_scalar('data_time', data_time.avg, epoch)

def _get_loss_info(meter, loss_name):
    #msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})'.format(name=loss_name.replace('_loss',''), meter=meter)
    msg = '{name}: ({meter.avg:.3e})'.format(name=loss_name.replace('_loss',''), meter=meter)
    return msg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
