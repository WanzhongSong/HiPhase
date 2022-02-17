import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib

from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage


import sys
sys.path.append("..")

from eval.dataset import dentalphase
from eval.HiPhase import Net
from eval.transform import Relabel, ToLabel, Colorize
from eval.iouEval import iouEval, getColorEntry

NUM_CLASSES = 18

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    ToTensor(),
])
target_transform_cityscapes = Compose([
    ToLabel(),
    Relabel(255, 17),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    loader = DataLoader(dentalphase(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)

        with torch.no_grad():
            outputs = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)
        filenameSave = filename[0].split("leftImg8bit/")[1]
        print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "order 0")
    print(iou_classes_str[1], "order 1")
    print(iou_classes_str[2], "order 2")
    print(iou_classes_str[3], "order 3")
    print(iou_classes_str[4], "order 4")
    print(iou_classes_str[5], "order 5")
    print(iou_classes_str[6], "order 6")
    print(iou_classes_str[7], "order 7")
    print(iou_classes_str[8], "order 8")
    print(iou_classes_str[9], "order 9")
    print(iou_classes_str[10], "order 10")
    print(iou_classes_str[11], "order 11")
    print(iou_classes_str[12], "order 12")
    print(iou_classes_str[13], "order 13")
    print(iou_classes_str[14], "order 14")
    print(iou_classes_str[15], "order 15")
    print(iou_classes_str[16], "order 16")
    print(iou_classes_str[17], "background and invalid")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadDir', default="D:/BaiduNetdiskDownload/HiPhase/code/save/HiPhase_experi_old/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="HiPhase.py")
    parser.add_argument('--subset', default="test")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="D:/BaiduNetdiskDownload/HiPhase/datasets/SCU-Phase-Ready/")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
