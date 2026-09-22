import numpy as np
from skimage.exposure import equalize_adapthist
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    """Retina dataset."""

    def __init__(self, imgs, gts, transform=None):
        """
        Arguments:
            imgs (list[array]): List with images.
            gts (list[array]): List with ground truths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = imgs
        self.gts = gts
        self.transform = transform

    def __len__(self):
        return np.shape(self.imgs)[0]

    def __getitem__(self, index):
        img = self.imgs[index, :, :]
        gt = self.gts[index, :, :]

        lo, hi = np.percentile(img, (5, 95))
        image_norm_percentile = (img - lo) / (hi - lo)
        sample = dict()
        sample["input"] = img
        sample["target"] = gt

        if self.transform is not None:
            transformed = self.transform(image=img, mask=gt)
            img = transformed["image"]
            gt = transformed["mask"]
            gt[gt != 0] = 1
            sample["input"] = img
            sample["target"] = gt

        return sample
