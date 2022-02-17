import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

import sys
sys.path.append("..")


from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage


from eval.dataset import dentalphase
from eval.HiPhase import Net
from eval.transform import Relabel, ToLabel, Colorize

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
    i =0
    time_train = []
    with torch.no_grad():
        for step, (images, labels, filename, filenameGt) in enumerate(loader):

            torch.cuda.synchronize()
            start_time = time.time()
            images = images.cuda()
            inputs = Variable(images)
            outputs = model(inputs)
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

            label = outputs[0].max(0)[1].byte().cpu().data

            if i != 0:  # first run always takes some time for setup
                fwt = time.time() - start_time
                time_train.append(fwt)
                print("Forward time per img (b=%d): %.6f (Mean: %.6f)" % (
                    args.batch_size, fwt / args.batch_size, sum(time_train) / len(time_train) / args.batch_size))

            time.sleep(1)  # to avoid overheating the GPU too much
            i += 1



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')
    parser.add_argument('--loadDir', default="D:/BaiduNetdiskDownload/HiPhase/code/save/HiPhase_experi_old/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="HiPhase.py")
    parser.add_argument('--subset', default="test")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="D:/BaiduNetdiskDownload/HiPhase/datasets/SCU-Phase-Ready/")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
