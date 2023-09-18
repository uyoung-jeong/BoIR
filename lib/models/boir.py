import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from .backbone import build_backbone
from .cid_module import ParamSampler
from .boir_module import MultiHeadWrapper, InstKptHead
from utils.torch_utils import _sigmoid

class BoIR(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg, is_train)
        self.multihead = MultiHeadWrapper(cfg, is_train) # multihead except instance-wise kpt head
        self.kpt_head = InstKptHead(cfg)
        self.param_sampler = ParamSampler(cfg, is_train)

        self.multi_heatmap_loss_weight = cfg.LOSS.MULTI_HEATMAP_LOSS_WEIGHT
        self.single_heatmap_loss_weight = cfg.LOSS.SINGLE_HEATMAP_LOSS_WEIGHT
        self.bbox_mask_loss_weight = cfg.LOSS.BBOX_MASK_LOSS_WEIGHT
        self.bbox_loss_weight = cfg.LOSS.BBOX_LOSS_WEIGHT

        self.remove_aux_head = cfg.TEST.REMOVE_AUX_HEAD

        # inference
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_index = cfg.DATASET.FLIP_INDEX
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL

        self.use_amp = cfg.TRAIN.AMP

        pretrained_file = cfg.TEST.MODEL_FILE
        if pretrained_file != '':
            transfer_dataset = cfg.TRAIN.TRANSFER_DATASET
            print("loading model from {}".format(pretrained_file))
            pretrained_dict = torch.load(pretrained_file)
            if transfer_dataset: # drop keypoint heads with different output size
                drop_keys = ['multihead.buk_head.head.1.weight', 'multihead.buk_head.head.1.bias',
                             'kpt_head.heatmap_conv.weight', 'kpt_head.heatmap_conv.bias']
                for drop_key in drop_keys:
                    pretrained_dict.pop(drop_key)
            self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, batch_inputs):
        if isinstance(batch_inputs, dict):
            images = batch_inputs['image'].to(self.device) # [b, 3, 512(h), 512(w)]
        else:
            images = [x['image'] for x in batch_inputs]
            images = torch.stack(images).to(self.device)
            if len(images.shape) == 5:
                images = images.squeeze(1)
        batch_size = images.size(0)

        if self.use_amp and self.training:
            with torch.cuda.amp.autocast():
                feats = self.backbone(images)
        else:
            feats = self.backbone(images)

        if self.training:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    multihead_out = self.multihead(feats, batch_inputs)
            else:
                multihead_out = self.multihead(feats, batch_inputs)
            multi_heatmap_loss = multihead_out['multi_heatmap_loss']
            center_map = multihead_out['center_map']
            emb = multihead_out['emb_map']
            bbox_map = multihead_out['bbox_map']

            losses = {}
            
            with torch.cuda.amp.autocast(enabled=False):
                sampler_dict = self.param_sampler(feats, emb, batch_inputs=batch_inputs, 
                                                  pred_multi_heatmap=center_map,
                                                  bbox_map = bbox_map)
                instances = sampler_dict['instances']
                bbox_loss = sampler_dict['bbox_loss']
                losses.update({'bbox_loss': bbox_loss * self.bbox_loss_weight})
                
                bbox_mask_loss = sampler_dict['bbox_mask_loss']
                losses.update({'bbox_mask_loss': bbox_mask_loss * self.bbox_mask_loss_weight})

            # limit max instances in training
            if 0 <= self.max_instances < len(instances['instance_param']):
                inds = torch.randperm(instances['instance_param'].size(0), device=self.device).long()
                for k, v in instances.items():
                    instances[k] = v[inds[:self.max_instances]]

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    single_heatmap_loss = self.kpt_head(feats, instances)
            else:
                single_heatmap_loss = self.kpt_head(feats, instances)

            losses.update({'multi_heatmap_loss': multi_heatmap_loss * self.multi_heatmap_loss_weight})
            losses.update({'single_heatmap_loss': single_heatmap_loss * self.single_heatmap_loss_weight})

            aux = {} # auxiliary info
            aux['emb_max'] = emb.max().detach()
            aux['emb_min'] = emb.min().detach()
            aux['emb_mean'] = emb.mean().detach()
            return losses, aux
        else:
            results = {}
            if self.flip_test:
                feats[1, :, :, :] = feats[1, :, :, :].flip([2])

            multihead_out = self.multihead(feats)
            pred_multi_heatmap = multihead_out['center_map']

            instances = self.param_sampler(feats,
                                            pred_multi_heatmap=pred_multi_heatmap)
                                            
            if instances is None: return results
            elif not 'instance_score' in instances: return results

            instance_heatmaps = self.kpt_head(feats, instances) # input feature is fixed to be backbone
            
            instance_scores = instances['instance_score']
            if self.flip_test:
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(instance_heatmaps, 2, dim=0)
                instance_heatmaps_flip = instance_heatmaps_flip[:, self.flip_index, :, :]
                instance_heatmaps = (instance_heatmaps + instance_heatmaps_flip) / 2.0
                instance_scores, instance_scores_flip = torch.chunk(instance_scores, 2, dim=0)

            num_people, num_keypoints, h, w = instance_heatmaps.size()
            center_pool = F.avg_pool2d(instance_heatmaps, self.center_pool_kernel, 1, (self.center_pool_kernel-1)//2)
            instance_heatmaps = (instance_heatmaps + center_pool) / 2.0
            nms_instance_heatmaps = instance_heatmaps.view(num_people, num_keypoints, -1)
            vals, inds = torch.max(nms_instance_heatmaps, dim=2)
            x, y = inds % w, (inds / w).long()
            # shift coords by 0.25
            x, y = self.adjust(x, y, instance_heatmaps)

            vals = vals * instance_scores.unsqueeze(1)
            poses = torch.stack((x, y, vals), dim=2)

            poses[:, :, :2] = poses[:, :, :2] * 4 + 2
            scores = torch.mean(poses[:, :, 2], dim=1)

            results.update({'poses': poses})
            results.update({'scores': scores})
            return results

    def adjust(self, res_x, res_y, heatmaps):
        n, k, h, w = heatmaps.size()#[2:]
        if hasattr(w, 'device'): # don't know why this happen
            w = w.to(res_x.device)
            h = h.to(res_x.device)

        x_l, x_r = (res_x - 1).clamp(min=0), (res_x + 1).clamp(max=w-1)
        y_t, y_b = (res_y + 1).clamp(max=h-1), (res_y - 1).clamp(min=0)
        n_inds = torch.arange(n, device=self.device)[:, None]
        k_inds = torch.arange(k, device=self.device)[None]

        px = torch.sign(heatmaps[n_inds, k_inds, res_y, x_r] - heatmaps[n_inds, k_inds, res_y, x_l])*0.25
        py = torch.sign(heatmaps[n_inds, k_inds, y_t, res_x] - heatmaps[n_inds, k_inds, y_b, res_x])*0.25

        res_x, res_y = res_x.float(), res_y.float()
        x_l, x_r = x_l.float(), x_r.float()
        y_b, y_t = y_b.float(), y_t.float()
        px = px*torch.sign(res_x-x_l)*torch.sign(x_r-res_x)
        py = py*torch.sign(res_y-y_b)*torch.sign(y_t-res_y)

        res_x = res_x.float() + px
        res_y = res_y.float() + py

        return res_x, res_y
