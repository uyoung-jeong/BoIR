import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math

from .loss import FocalLoss
from .emb_loss import AELoss
from utils.torch_utils import _sigmoid
from .att_module import compute_locations
from .bbox_mask_loss import BboxMaskLoss, BboxLoss, dist2bbox

# training: compute embedding loss and form instance parameter
# inference: return instance parameter
class ParamSampler(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.hm_size = cfg.DATASET.OUTPUT_SIZE

        self.emb_dim = cfg.MODEL.NECK.IN_CHANNELS
        self.contrastive_loss = AELoss(device=self.device, beta=cfg.LOSS.AE_BETA)
        
        self.bbox_mask_loss = BboxMaskLoss(cfg, self.contrastive_loss)
        self.bbox_loss = BboxLoss()

        self.remove_aux_head = cfg.TEST.REMOVE_AUX_HEAD
        self.use_aux_head = is_train or not self.remove_aux_head

        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2

    def _sample_feats(self, features, pos_ind):
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]].permute(1, 0)
        return feats

    def _sample_param(self, features, emb, instance_coord, ind, sample_option='loss'):
        condition = 'emb' if sample_option=='loss' else 'backbone'
        if condition == 'backbone':
            instance_param = self._sample_feats(features[ind], instance_coord)
        else:
            if emb is None and self.training:
                raise RuntimeError(f'emb is None while param sample target is {condition}')
        return instance_param

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > self.pool_thre1:
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > self.pool_thre2:
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm
    
    def forward(self, features, emb=None, batch_inputs=None, pred_multi_heatmap=None, bbox_map=None):
        if self.training:
            total_instances = 0
            target_dtype = torch.float32
            bbox_loss = torch.zeros(1, dtype=target_dtype, device=features.device).sum()
            instances = defaultdict(list)
            batch_size = features.size(0)
            H,W = features.shape[-2:]
            batch_gt_bboxs = [[] for _ in range(batch_size)]
            total_extended_instances = 0
            for i in range(batch_size): # batch size loop
                if isinstance(batch_inputs, dict): # DP input
                    num_inst = batch_inputs['num_inst'][i]
                    if num_inst < 1 : continue
                    instance_coord = batch_inputs['instance_coord'][i][:num_inst].to(self.device) # [N, 2]
                    instance_heatmap = batch_inputs['instance_heatmap'][i][:num_inst].to(self.device)
                    instance_mask = batch_inputs['instance_mask'][i][:num_inst].to(self.device)
                    bboxs = batch_inputs['bbox'][i][:num_inst].to(self.device)
                    gt_multi_mask = batch_inputs['multi_mask'].to(self.device)
                else: # DDP input
                    if 'instance_coord' not in batch_inputs[i]: continue
                    instance_coord = batch_inputs[i]['instance_coord'].to(self.device)
                    instance_heatmap = batch_inputs[i]['instance_heatmap'].to(self.device)
                    instance_mask = batch_inputs[i]['instance_mask'].to(self.device)
                    bboxs = batch_inputs[i]['bbox'].to(self.device)
                    gt_multi_mask = [x['multi_mask'].unsqueeze(0).to(self.device) for x in batch_inputs]
                    gt_multi_mask = torch.cat(gt_multi_mask, dim=0)
                batch_gt_bboxs[i] = bboxs
                num_inst = instance_coord.size(0)

                instance_imgid = i * torch.ones(num_inst, dtype=torch.long).to(self.device)

                # param sampling
                instance_param = self._sample_param(features, emb, instance_coord, ind=i, sample_option='inst')
                total_instances += num_inst

                # bbox loss
                min_x, max_x = torch.clamp(instance_coord[:,1]-1,0), torch.clamp(instance_coord[:,1]+1,0,W-1)
                min_y, max_y = torch.clamp(instance_coord[:,0]-1,0), torch.clamp(instance_coord[:,0]+1,0,H-1)

                pred_bboxs = []
                gt_ext_bboxs = []
                for ii in range(num_inst):
                    xrange = torch.arange(min_x[ii], max_x[ii]+1, device=self.device)
                    yrange = torch.arange(min_y[ii], max_y[ii]+1, device=self.device)
                    xcoord, ycoord = torch.meshgrid(xrange, yrange)
                    extended_coord = torch.cat((ycoord.flatten().unsqueeze(1),xcoord.flatten().unsqueeze(1)),dim=1)
                    total_extended_instances += extended_coord.size(0)
                    pred_bbox_ltrb = self._sample_feats(bbox_map[i], extended_coord)
                    pred_bbox = dist2bbox(pred_bbox_ltrb, extended_coord, box_format='xywh')
                    pred_bbox = pred_bbox.type(torch.float32)
                    extended_bboxs = bboxs[ii].expand(extended_coord.size(0), -1)
                    pred_bboxs.append(pred_bbox)

                    gt_ext_bboxs.append(extended_bboxs)
                pred_bboxs = torch.cat(pred_bboxs, dim=0)

                gt_ext_bboxs = torch.cat(gt_ext_bboxs, dim=0)
                bbox_loss = bbox_loss + self.bbox_loss(pred_bboxs, gt_ext_bboxs)

                instances['instance_coord'].append(instance_coord)
                instances['instance_imgid'].append(instance_imgid)
                instances['instance_param'].append(instance_param)
                instances['instance_heatmap'].append(instance_heatmap)
                instances['instance_mask'].append(instance_mask)
                instance_scores = self._sample_feats(pred_multi_heatmap[i,-1].unsqueeze(0), instance_coord)
                instances['instance_score'].append(instance_scores)
            
            for k, v in instances.items():
                instances[k] = torch.cat(v, dim=0)

            return_dict = {'instances': instances}
            
            emb = emb.type(torch.float32)
            bbox_mask_loss = self.bbox_mask_loss(emb, batch_gt_bboxs)
            return_dict['bbox_mask_loss'] = bbox_mask_loss/total_instances
            return_dict['bbox_loss'] = bbox_loss/total_extended_instances

            return return_dict
        else:
            instances = {}
            H, W = pred_multi_heatmap.shape[-2:]
            if self.flip_test:
                center_heatmap = pred_multi_heatmap[:, -1, :, :].mean(dim=0, keepdim=True)
            else:
                center_heatmap = pred_multi_heatmap[:, -1, :, :]

            center_pool = F.avg_pool2d(center_heatmap, self.center_pool_kernel, 1, (self.center_pool_kernel-1)//2)
            center_heatmap = (center_heatmap + center_pool) / 2.0
            maxm = self.hierarchical_pool(center_heatmap)
            maxm = torch.eq(maxm, center_heatmap).float()
            center_heatmap = center_heatmap * maxm
            scores = center_heatmap.view(-1)
            scores, pos_ind = scores.topk(self.max_proposals, dim=0)
            select_ind = (scores > (self.keypoint_thre)).nonzero()

            if len(select_ind) > 0:
                scores = scores[select_ind].squeeze(1)
                pos_ind = pos_ind[select_ind].squeeze(1)
                x = pos_ind % W
                y = (pos_ind / W).long()
                instance_coord = torch.stack((y, x), dim=1)
                instance_param = self._sample_param(features, emb, instance_coord, ind=0, sample_option='inst')
                instance_imgid = torch.zeros(instance_coord.size(0), dtype=torch.long).to(features.device)

                if self.flip_test:
                    instance_param_flip = self._sample_param(features, emb, instance_coord, ind=1, sample_option='inst')
                    instance_imgid_flip = torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                    instance_coord_cat = torch.cat((instance_coord, instance_coord), dim=0)
                    instance_param = torch.cat((instance_param, instance_param_flip), dim=0)
                    instance_imgid = torch.cat((instance_imgid, instance_imgid_flip), dim=0)
                    scores = torch.cat((scores, scores), dim=0)
                else:
                    instance_coord_cat = instance_coord

                instances['instance_coord'] = instance_coord_cat
                instances['instance_imgid'] = instance_imgid
                instances['instance_param'] = instance_param
                instances['instance_score'] = scores
            return instances

class ChannelAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtten, self).__init__()
        atn = [nn.Linear(in_channels, out_channels)]
        atn.append(nn.LayerNorm(out_channels))
        self.atn = nn.Sequential(*atn)

    def forward(self, global_features, instance_params):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1)
        ch_atn = global_features * instance_params.expand_as(global_features)
        return ch_atn

class SpatialAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAtten, self).__init__()
        atn = [nn.Linear(in_channels, out_channels)]
        atn.append(nn.LayerNorm(out_channels))
        self.atn = nn.Sequential(*atn)

        self.feat_stride = 4
        conv_in = 3
        self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2)
        self.nonlinear = nn.InstanceNorm2d(1)

    def forward(self, global_features, instance_params, instance_inds):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params)
        instance_params = instance_params.reshape(B, C, 1, 1)
        feats = global_features * instance_params.expand_as(global_features)
        input_feats = torch.sum(feats, dim=1, keepdim=True)
        locations = compute_locations(global_features.size(2), global_features.size(3), stride=1, device=global_features.device)
        n_inst = instance_inds.size(0)
        instance_locations = torch.flip(instance_inds, [1])
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = (relative_coords / 32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)
        input_feats = torch.cat((input_feats, relative_coords), dim=1)
        mask = self.conv(input_feats)
        mask = self.nonlinear(mask)

        sp_atn = global_features * mask
        return sp_atn
