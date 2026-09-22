import torch


def dice(prediction, target, threshold):
    # prediction[prediction > threshold] = 1
    # prediction[prediction != 1] = 0
    eps = 1e-7
    intersection = (prediction * target).sum(dim=(1, 2))
    union = prediction.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice
