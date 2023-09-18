import torch
from torch import nn
import torch.nn.functional as F

import math

from .emb_loss import l2

# https://github.com/meituan/YOLOv6/blob/6b9f5f4ea3185496b5f62a934c3c8f2d095c0318/yolov6/utils/general.py
def bbox2dist(anchor_points, bbox, reg_max):
    '''Transform bbox(xyxy) to dist(ltrb).'''
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist

# https://github.com/meituan/YOLOv6/blob/6b9f5f4ea3185496b5f62a934c3c8f2d095c0318/yolov6/utils/general.py
# distance : [N, 4]
# anchor_points: [N, 2]
def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        #c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        #bbox = torch.cat([c_xy, wh], -1)
        bbox = torch.cat([x1y1, wh], -1)
    return bbox

# https://github.com/meituan/YOLOv6/blob/6b9f5f4ea3185496b5f62a934c3c8f2d095c0318/yolov6/utils/figure_iou.py
class IOUloss:
    """ Calculate IoU loss.
    """
    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == 'xywh':
                b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
                b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
                b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
                b2_y1, b2_y2 = box2[1], box2[1] + box2[3]
        else:
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

            elif self.box_format == 'xywh':
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x2 = b1_x1 + b1_w
                b1_y2 = b1_y1 + b1_h
                b2_x2 = b2_x1 + b2_w
                b2_y2 = b2_y1 + b2_h
        
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

# https://github.com/meituan/YOLOv6/blob/6b9f5f4ea3185496b5f62a934c3c8f2d095c0318/yolov6/models/losses/loss.py
class BboxLoss(nn.Module):
    def __init__(self, reg_max=0, iou_type='ciou'):
        super(BboxLoss, self).__init__()
        self.iou_loss = IOUloss(box_format='xywh', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max # 0

    # pred_bboxes: [N, 4]
    # target_bboxes: [N, 4]
    def forward(self, pred_bboxes, target_bboxes):
        pos_mask = torch.abs(target_bboxes).sum(dim=1) > 0
        num_pos = pos_mask.sum()
        if num_pos > 0:
            pred_bboxes_pos = pred_bboxes
            target_bboxes_pos = target_bboxes

            loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos).sum()
        else:
            loss_iou = torch.zeros(1).float().sum().to(pred_bboxes.device)

        return loss_iou


# bbox-derived mask loss
class BboxMaskLoss(nn.Module):
    # loss_func: pre-initialized loss object
    def __init__(self, cfg, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.dist = l2
        self.ae_beta = cfg.LOSS.AE_BETA
        self.bg_sample_level = cfg.LOSS.BBOX_MASK_LOSS_BG_SAMPLE_LEVEL

        out_size = cfg.DATASET.OUTPUT_SIZE
        mask = torch.zeros(out_size, out_size, dtype=torch.float32)
        self.register_buffer("mask", mask, persistent=False) # pre-initialize mask
        self.eps = 1.0e-5

    # emb: [B, D x H x W]
    # bboxs: B x [n x 4]. xywh format
    def forward(self, emb, bboxs):
        B,D,H,W = emb.shape
        device = emb.device
        mask = self.mask

        inst_wise_loss = torch.zeros(1, dtype=emb.dtype, device=emb.device).sum()
        cross_inst_loss = torch.zeros(1, dtype=emb.dtype, device=emb.device).sum()
        batch_inst_params = []

        emb = F.normalize(emb, dim=1)
        for bi in range(B):
            e = emb[bi]
            bbox_bi = bboxs[bi]
            N = len(bbox_bi)
            inst_params = []
            if self.bg_sample_level > 0:
                mask[:,:] = 0
            
            for ni in range(N):
                # make mask from bbox
                if self.bg_sample_level==0:
                    mask[:,:] = 0
                bbox = bbox_bi[ni]
                x1 = torch.clamp(bbox[0].int()-1, 0, W-1).long()
                x2 = torch.clamp((bbox[0]+bbox[2]).int()+1, 0, W-1).long()
                y1 = torch.clamp(bbox[1].int()-1, 0, H-1).long()
                y2 = torch.clamp((bbox[1]+bbox[3]).int()+1, 0, H-1).long()

                if x2-x1<2 or y2-y1<2: # too small
                    continue

                mask[y1:y2,x1:x2] = 1.0

                # compute similarity matrix
                center_x = torch.div((bbox[0]+bbox[2]),2, rounding_mode='floor')
                center_y = torch.div((bbox[1]+bbox[3]),2, rounding_mode='floor')
                center_param = e[:,center_y.long(),center_x.long()].unsqueeze(0)
                e_partial = e[:,y1:y2,x1:x2]
                _h, _w = e_partial.shape[-2:]
                center_param_reshape = center_param[0,:,None,None]
                e_partial = e_partial.detach()
                center_param_reshape = center_param_reshape.detach()
                inst_params.append(center_param)
                dist = (center_param_reshape - e_partial) ** 2
                _sim = torch.exp(-self.ae_beta * dist.sum(dim=0))
                _sim = _sim / max(_sim.max().item(), 1.0) # bound by 1.0

                # pull loss
                pos_param = center_param
                sim_sum = _sim.sum()
                if sim_sum < self.eps: # too small similar region
                    mean_param = center_param
                else:
                    mean_param = e[:,y1:y2,x1:x2] * _sim[None,:,:] / sim_sum
                    mean_param = mean_param.sum(dim=-1).sum(dim=-1).unsqueeze(0) # reduce H,W dim
                    pos_param = torch.cat((center_param, mean_param), dim=0)
                
                # out-bbox push loss
                if self.bg_sample_level==0:
                    mask_sum = mask.sum()
                    neg_param = e * (1-mask[None,:,:])
                    neg_param = neg_param.sum(dim=-1).sum(dim=-1).unsqueeze(0)/(H*W-mask_sum)

                    inst_input = [pos_param, neg_param]
                    inst_wise_loss = inst_wise_loss + self.loss_func(inst_input)
                    
            bg_params = None
            if self.bg_sample_level>0: # inst-aware background sampling
                e_patch = e.reshape(1,e.size(0),-1)
                mask_patch = mask.reshape(1,-1)
                bg_mask_patch = 1-mask_patch
                
                valid_patch_idxs = bg_mask_patch.sum(dim=-1) > 0
                if valid_patch_idxs.sum().item() > 0:
                    e_patch = e_patch[valid_patch_idxs]
                    bg_mask_patch = bg_mask_patch[valid_patch_idxs]

                    e_patch_bg = e_patch * bg_mask_patch[:,None,:]
                    bg_params = e_patch_bg.sum(dim=-1) / bg_mask_patch.sum(dim=-1)[:,None]

            # multi-instance push loss
            if len(inst_params)>0:
                inst_params = torch.cat(inst_params, dim=0)
                if bg_params is not None:
                    inst_params = torch.cat((inst_params, bg_params), dim=0)
                if len(inst_params)>1:
                    cross_inst_loss = cross_inst_loss + self.loss_func(inst_params)

        total_loss = inst_wise_loss + cross_inst_loss
            
        return total_loss
    
