import numpy as np
import torch
from PIL import Image

def colormap_dentalphase(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([139, 119, 101])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([0, 139, 139])
    cmap[3, :] = np.array([125, 38, 205])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])
    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220,  0])
    cmap[8, :] = np.array([255, 250, 205])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])
    cmap[11, :] = np.array([0, 255, 255])
    cmap[12, :] = np.array([255,  0,  0])
    cmap[13, :] = np.array([205, 129, 98])
    cmap[14, :] = np.array([0,  191, 255])
    cmap[15, :] = np.array([255, 62, 150])
    cmap[16, :] = np.array([139, 0, 0])
    cmap[17, :] = np.array([255, 255, 255])

    return cmap


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=22):
        self.cmap = colormap_dentalphase(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
