import torch
import torch.nn as nn
import torch.nn.functional as F


"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


""" Deep Supervision Loss"""

#
# def DeepSupervisionLoss(pred, gt):
#     d0, d1, d2, d3, d4 = pred[0:]
#
#     criterion = BceDiceLoss()
#
#     loss0 = criterion(d0, gt)
#     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
#     loss1 = criterion(d1, gt)
#     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
#     loss2 = criterion(d2, gt)
#     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
#     loss3 = criterion(d3, gt)
#     gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
#     loss4 = criterion(d4, gt)
#
#     return loss0 + loss1 + loss2 + loss3 + loss4


def DeepSupervisionLoss(pred, gt):
    # Resize the ground truth to match the prediction size (7x7)
    gt_resized = F.interpolate(gt, size=pred.size()[2:], mode='bilinear', align_corners=True)

    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(pred, gt_resized)

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        # print(inputs.shape)
        # print(targets.shape)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

