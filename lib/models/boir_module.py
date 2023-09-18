import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math

from .loss import FocalLoss
from .att_module import ASPP
from .cid_module import ChannelAtten, SpatialAtten
from .backbone import BasicBlock
from utils.torch_utils import _sigmoid, NormConv

class Head(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels

        downsample = NormConv(self.in_channels, self.channels, kernel_size=1, stride=1, padding=0, dilation=1,
                            bias=False, norm='batch')

        head_layers = [BasicBlock(self.in_channels, self.channels, stride=1, downsample=downsample)]
        head_layers.append(nn.Conv2d(self.channels, self.out_channels, 3, 1, 1, 1, bias=True))
        self.head = nn.Sequential(*head_layers)

    def forward(self, feat):
        out = self.head(feat)
        return out

class MultiHeadWrapper(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # build neck
        modules = cfg.MODEL.NECK.MODULES
        num_layers = len(modules.NAME)
        self.in_channels = cfg.MODEL.NECK.IN_CHANNELS
        in_channels = self.in_channels
        out_channels = in_channels
        layers = []
        for l_i in range(num_layers):
            out_channels = modules.CHANNELS[l_i]
            atrous_rates = modules.ATROUS_RATES
            middle_channels = modules.MIDDLE_CHANNELS[l_i]
            module = ASPP(in_channels=in_channels, atrous_rates=atrous_rates, channels=middle_channels, out_channels=out_channels)
            layers.append(module)
            in_channels = out_channels
        self.neck = nn.Sequential(*layers)

        # build heads
        self.center_head = Head(cfg.MODEL.CenterHead.IN_CHANNELS,
                                cfg.MODEL.CenterHead.CHANNELS,
                                cfg.MODEL.CenterHead.OUT_CHANNELS)
        
        self.remove_aux_head = cfg.TEST.REMOVE_AUX_HEAD
        self.use_aux_head = is_train or not self.remove_aux_head
        if self.use_aux_head:
            self.buk_head = Head(cfg.MODEL.BUKHead.IN_CHANNELS,
                                    cfg.MODEL.BUKHead.CHANNELS,
                                    cfg.MODEL.BUKHead.OUT_CHANNELS)
            self.emb_head = Head(cfg.MODEL.EmbHead.IN_CHANNELS,
                                    cfg.MODEL.EmbHead.CHANNELS,
                                    cfg.MODEL.EmbHead.OUT_CHANNELS)
                        
            # loss
            self.heatmap_loss = FocalLoss()
            
        # initialize weights
        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, bias_value)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if self.use_aux_head:
            self.bbox_head = Head(cfg.MODEL.BboxHead.IN_CHANNELS,
                                cfg.MODEL.BboxHead.CHANNELS,
                                cfg.MODEL.BboxHead.OUT_CHANNELS)

            # initialize weights
            for module in self.bbox_head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.constant_(module.weight, 0.)
                    for name, _ in module.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(module.bias, 1.0)
            
        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2

    def forward(self, features, batch_inputs=None):
        neck_features = self.neck(features)
        center = self.center_head(neck_features)

        return_dict = {}

        if self.training:
            buk = self.buk_head(neck_features)
            emb = self.emb_head(neck_features)

        #return_dict = {'center_map':center}
        if self.training:
            bbox = self.bbox_head(neck_features)
            return_dict['bbox_map'] = bbox

            with torch.cuda.amp.autocast(enabled=False):
                if isinstance(batch_inputs, dict):
                    gt_multi_heatmap = batch_inputs['multi_heatmap'].to(self.device)
                    gt_multi_mask = batch_inputs['multi_mask'].to(self.device)
                else:
                    gt_multi_heatmap = [x['multi_heatmap'].unsqueeze(0).to(self.device) for x in batch_inputs]
                    gt_multi_heatmap = torch.cat(gt_multi_heatmap, dim=0)
                    gt_multi_mask = [x['multi_mask'].unsqueeze(0).to(self.device) for x in batch_inputs]
                    gt_multi_mask = torch.cat(gt_multi_mask, dim=0)

                hm_cat = torch.cat((buk, center), dim=1)

                pred_multi_heatmap = _sigmoid(hm_cat.type(torch.float32))
                multi_heatmap_loss = self.heatmap_loss(pred_multi_heatmap, gt_multi_heatmap, gt_multi_mask)
                
                return_dict['center_map'] = pred_multi_heatmap[:,-1:]
                return_dict['multi_heatmap_loss'] = multi_heatmap_loss
                return_dict['emb_map'] = emb
        else:
            with torch.cuda.amp.autocast(enabled=False):
                center = _sigmoid(center.type(torch.float32))
                return_dict['center_map'] = center
        return return_dict

# instance-wise keypoint regressor
class InstKptHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.InstKptHead.IN_CHANNELS
        self.channels = cfg.MODEL.InstKptHead.CHANNELS
        self.out_channels = cfg.MODEL.InstKptHead.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints
        self.prior_prob = cfg.MODEL.BIAS_PROB

        self.conv_down = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.c_attn = ChannelAtten(self.in_channels, self.channels)
        self.s_attn = SpatialAtten(self.in_channels, self.channels)

        self.fuse_attn = nn.Sequential(nn.Conv2d(self.channels*2, self.channels, 1, 1, 0),
                                    nn.InstanceNorm2d(self.channels))

        self.heatmap_conv = nn.Conv2d(self.channels, self.out_channels, 1, 1, 0)

        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD

        self.heatmap_loss = FocalLoss()

        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
                        m.bias.data.fill_(bias_value)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    # requires instance parameter to be embedding
    def forward(self, features, instances):
        instance_imgids = instances['instance_imgid']
        instance_params = instances['instance_param']

        global_features = self.conv_down(features)
        instance_features = global_features[instance_imgids]
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params, instances['instance_coord'])
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats), dim=1)
        cond_instance_feats = self.fuse_attn(cond_instance_feats)
        cond_instance_feats = F.relu(cond_instance_feats)

        hm_out = self.heatmap_conv(cond_instance_feats)
        with torch.cuda.amp.autocast(enabled=False):
            pred_instance_heatmaps = _sigmoid(hm_out.type(torch.float32))
            if self.training:
                gt_instance_heatmaps = instances['instance_heatmap']
                gt_instance_masks = instances['instance_mask']
                single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps, gt_instance_heatmaps, gt_instance_masks)
                return single_heatmap_loss
            else:
                return pred_instance_heatmaps

