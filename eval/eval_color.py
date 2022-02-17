import numpy as np
import torch
import os
import importlib

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

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")

    model.eval()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        dentalphase(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data

        label_color = Colorize()(label.unsqueeze(0))
        filenameSave = "./HiPhase_expri_color/" + filename[0].split("leftImg8bit/")[1]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        label_save = ToPILImage()(label_color)
        label_save.save(filenameSave)
        print(step, filenameSave)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadDir', default="D:/pytorchproject/HiPhase/save/HiPhase_experi/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="HiPhase.py")
    parser.add_argument('--subset', default="test")  # can be val, test, train, demoSequence
    parser.add_argument('--datadir', default="D:/pytorchproject/HiPhase/datasets/SCU-Phase/")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    main(parser.parse_args())

