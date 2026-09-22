import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(
        self,
    ):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        ignore_background = False
        if ignore_background:
            target = target[:, 1:, :, :]
            input = input[:, 1:, :, :]
        # axes = (2, 3)
        axes = (0, 2, 3)
        eps = 1e-7

        num = 2 * torch.sum(target * input, axis=axes)
        denom = torch.sum(target, axis=axes) + torch.sum(input, axis=axes) + eps
        score = num / denom
        loss = 1 - score.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = -(
            target * torch.pow((1 - input), self.gamma) * torch.log(input)
            + (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input)
        )
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
