import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask=None):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
              pred (batch x c x h x w)
              gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        if mask is not None:
            pos_inds = pos_inds * mask
            neg_inds = neg_inds * mask

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss
