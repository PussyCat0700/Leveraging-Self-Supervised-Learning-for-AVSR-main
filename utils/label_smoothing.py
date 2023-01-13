import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SmoothCTCLoss(_Loss):
    def __init__(self, num_classes, blank=0, weight=0.01):
        super().__init__(reduction='mean')
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')  #KL-divergence

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)  #log_probs(155,64,40)  input_lengths:(64,) tensor([ 80,  80,  88, 151,  80,  80,  80, 108, ...) 加在一起是6099  这里最大长度是155
        #targets(3384,)  target_lengths:(64,)   tensor([ 12,  58,  65,  87,  15,  35,  20,  80,  30,  52,  说明input和target长度不一致
        kl_inp = log_probs.transpose(0, 1)  #(8,152,40)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes) # (8,152,40)
        kldiv_loss = self.kldiv(kl_inp, kl_tar) # 12,2485
        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss #11.3492
        return loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon: float = 0.01, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]   #preds (64,140,40) target(64,140)
        log_preds = F.log_softmax(preds.reshape(-1, n), dim=-1) #(736,40)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target.long().reshape(-1), ignore_index=0, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
