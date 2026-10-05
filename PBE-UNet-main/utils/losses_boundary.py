import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.seg_loss = BCEDiceLoss()

        self.alpha = alpha

    def forward(self, seg_pred, boundary_preds, seg_gt, boundary_gt):
        loss_seg = self.seg_loss(seg_pred, seg_gt)

        b_look = []

        bceloss_boundary = 0
        for pred in boundary_preds:
            pred = F.interpolate(pred, boundary_gt.shape[2:], mode='bilinear')
            tempbce = F.binary_cross_entropy_with_logits(pred, boundary_gt)
            bceloss_boundary += tempbce

            b_look.append(tempbce)

        return loss_seg + self.alpha * bceloss_boundary, b_look



