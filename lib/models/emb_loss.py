import torch
from torch import nn
import torch.nn.functional as F
from utils.torch_utils import _sigmoid

def l2(x,y,dim=None):
    #return (x-y)**2
    return F.mse_loss(x,y,reduction='none')

# https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/core/loss.py
class AELoss(nn.Module):
    def __init__(self, pull_factor=1.0, push_factor=1.0, device='cuda', beta=1.0):
        super(AELoss, self).__init__()
        self.dist = l2
        self.pull_factor = pull_factor
        self.push_factor = push_factor
        self.device = device
        self.beta = beta

    def forward(self, feats):
        """
        embeds: N x [M, D]
        N: # instances
        M: # positive samples
        D: embedding dim
        """
        pull = torch.zeros(1, dtype=torch.float32, device=self.device).sum()
        push = torch.zeros(1, dtype=torch.float32, device=self.device).sum()
        N = len(feats)
        assert N > 0
        D = feats[0].shape[-1]

        if len(feats[0].shape)==2: # keypoint supervision
            inst_embeds = []
            for n_i in range(N): # pull
                feat = feats[n_i].clone()
                feat_mean = feat.mean(dim=0, keepdim=True)
                pull = pull + torch.sum(self.dist(feat, feat_mean.expand(feat.shape[0],-1), dim=2))
                inst_embeds.append(feat_mean)
            inst_embeds = torch.cat(inst_embeds, dim=0) # [N,D]
        else:
            inst_embeds = feats
        pull = pull / D # no division by N, since it is divided outside the loss function

        if N > 1: # push
            diff = self.dist(inst_embeds[:,None,:].expand(-1,N,-1), inst_embeds[None,:,:].expand(N,-1,-1), dim=2)
            if diff.dim() == 3:
                diff = diff.mean(dim=-1)
            diff.diagonal().zero_() # fill diagonal elements with 0
            push = torch.clamp(torch.exp(-self.beta * diff).sum(), min=0.0)
            push = 0.5 * push / (N - 1) # no division by N, since it is divided outside the loss function

        return pull * self.pull_factor + push * self.push_factor
