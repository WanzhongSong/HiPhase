import os
import time
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import importlib
import cv2

from PIL import Image, ImageOps
from argparse import ArgumentParser
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import sys
sys.path.append("..")

from train.dataset import dentalphase
from train.transform import Relabel, ToLabel
from shutil import copyfile

NUM_CLASSES = 18

image_transform = ToPILImage()
class MyCoTransform(object):
    def __init__(self, enc):
        self.enc=enc
        pass
    def __call__(self, input, target):
        input = ToTensor()(input)
        if (self.enc):
            target = Resize([128,128], Image.NEAREST)(target)  #only in encoder-training
        target = ToLabel()(target)
        target = Relabel(255, 17)(target) #0-17

        return input, target

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model, enc=False):

    best_acc = 1

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.8400409575571
        weight[1] = 2.2152681284429
        weight[2] = 2.1354314373345
        weight[3] = 2.0080983569053
        weight[4] = 1.8559719699084
        weight[5] = 1.7713645889141
        weight[6] = 1.6665506268278
        weight[7] = 1.5584328705194
        weight[8] = 1.4978918887779
        weight[9] = 1.5236978199959
        weight[10] = 1.5143028604772
        weight[11] = 1.5318865317400
        weight[12] = 1.5648204002985
        weight[13] = 1.6261118419041
        weight[14] = 1.6977949028934
        weight[15] = 1.8414715725942
        weight[16] = 5.9995884343957
        weight[17] = 0.1155389693794
    else:
        weight[0] = 2.8400409575571
        weight[1] = 2.2152681284429
        weight[2] = 2.1354314373345
        weight[3] = 2.0080983569053
        weight[4] = 1.8559719699084
        weight[5] = 1.7713645889141
        weight[6] = 1.6665506268278
        weight[7] = 1.5584328705194
        weight[8] = 1.4978918887779
        weight[9] = 1.5236978199959
        weight[10] = 1.5143028604772
        weight[11] = 1.5318865317400
        weight[12] = 1.5648204002985
        weight[13] = 1.6261118419041
        weight[14] = 1.6977949028934
        weight[15] = 1.8414715725942
        weight[16] = 5.9995884343957
        weight[17] = 0.1155389693794

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(enc)
    co_transform_val = MyCoTransform(enc)
    dataset_train = dentalphase(args.datadir, co_transform, 'train')
    dataset_val = dentalphase(args.datadir, co_transform_val, 'val')
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.cuda()
    loss_func = CrossEntropyLoss2d(weight)
    print(type(loss_func))

    savedir = f'../save/{args.savedir}'
    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):    #do not add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tVal-loss\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    scaler = GradScaler()

    start_epoch = 1
    if args.resume:
        # Must load weights, optimizer, epoch, scaler and best value.
        if args.resumeencoder:
            if enc:
                print("resume-encoder")
                filenameCheckpointR = savedir + '/checkpoint_enc.pth.tar'
                assert os.path.exists(filenameCheckpointR), "Error: resumeencoder option was used but checkpoint was not found in folder"
                checkpoint = torch.load(filenameCheckpointR)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
                best_acc = checkpoint['best_acc']
            else:
                print("decoder training from epoch 1 with encoder_best.pth model")

        else:
            if enc:
                print("resume-whole")
                start_epoch = args.num_epochs + 10
            else:
                filenameCheckpointR = savedir + '/checkpoint.pth.tar'
                assert os.path.exists(filenameCheckpointR), "Error: resume option was used but checkpoint was not found in folder"
                checkpoint = torch.load(filenameCheckpointR)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
                best_acc = checkpoint['best_acc']
                print("=> Loaded whole checkpoint at epoch {})".format(checkpoint['epoch']))

    lambda1 = lambda epoch: pow((1-((epoch - 1)/args.num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)
        epoch_loss = []
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs, only_encode=enc)
                loss = loss_func(outputs, targets[:, 0])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        with torch.no_grad():
            for step, (images, labels) in enumerate(loader_val):
                start_time = time.time()
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                inputs = Variable(images)
                targets = Variable(labels)
                outputs = model(inputs, only_encode=enc)
                loss = loss_func(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())

                time_val.append(time.time() - start_time)

                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

            average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        lossVal = average_epoch_loss_val
        # remember best valloss and save checkpoint
        current_acc = lossVal
        is_best = current_acc < best_acc
        best_acc = min(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'scaler': scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if epoch > (args.num_epochs - 10): #save back models
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-Loss= %.4f" % (epoch, lossVal))
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-Loss= %.4f" % (epoch, lossVal))

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss)
        #Epoch		Train-loss		Val-loss		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, usedLr))

    if enc:
        filenameCheckpointE = savedir + '/model_best_enc.pth.tar'
        assert os.path.exists(filenameCheckpointE), "Error: resume option was used but checkpoint was not found in folder"
        checkpointE = torch.load(filenameCheckpointE)
        model.load_state_dict(checkpointE['state_dict'])
        print("=> From best checkpoint at epoch {} - 1)".format(checkpointE['epoch']))
        return(model)
    else:
        return(model)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    #Train
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
    print("========== DECODER TRAINING ===========")
    Enc = next(model.children()).encoder
    model = model_file.HiPhaseNet(NUM_CLASSES, encoder=Enc)  # Add decoder to encoder
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    model = train(args, model, False)   #Train whole
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="HiPhase")
    parser.add_argument('--datadir', default="D:/BaiduNetdiskDownload/HiPhase/datasets/SCU-Phase-Ready/")
    parser.add_argument('--num-epochs', type=int, default=120)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-loss', type=int, default=100)
    parser.add_argument('--savedir',default="HiPhase_experi/")
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resumeencoder', action='store_true') #resume from encoder
    main(parser.parse_args())
