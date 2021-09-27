import time
import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable

from dataset import HDR_Dataset, HDR_DataLoader_pre
from model import Deep_SR_HDR
from tqdm import trange
import numpy as np

import torch.nn.functional as F


torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=4, help='batch size')
parser.add_argument(
    '--train', '-f', default="./dataset_full/Train/TrainSequence.h5", type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=20000, help='max epochs')
parser.add_argument('--scale_factor', type=int, default=4, help='image height')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')

parser.add_argument('--nReslayer', type=int, default=3, help='nReslayer of RB')
parser.add_argument('--nDenselayer', type=int, default=4, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--filters_in', type=int, default=6, help='number of channels in')
parser.add_argument('--filters_out', type=int, default=3, help='number of channels out')
parser.add_argument('--groups', type=int, default=8, help='number of groups')
args = parser.parse_args()

if args.scale_factor == 2:
    crop_size = [512, 512]
else:
    crop_size = [1024, 768]

train_set = HDR_Dataset(datadir=args.train, crop_size=crop_size, scale_factor=args.scale_factor)

train_loader = HDR_DataLoader_pre(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

sr_hdr = Deep_SR_HDR(args=args).cuda()
solver = optim.Adam(
    [
        {
            'params': sr_hdr.parameters()
        },
    ],
    lr=args.lr)

MU = 5000.0


def tonemap(images):  # input/output 0~1
    return torch.log(1.0 + MU * images) / np.log(1.0 + MU)


def resume(epoch=None, scale=2):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    sr_hdr.load_state_dict(
        torch.load('checkpoint/sr_hdr_x{}_{}_{:08d}.pth'.format(scale, s, epoch)))


def save(index, epoch=True, scale=2):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(sr_hdr.state_dict(), 'checkpoint/sr_hdr_x{}_{}_{:08d}.pth'.format(scale, s, index))


scheduler = LS.MultiStepLR(solver, milestones=[18000], gamma=0.1)

# # x2 need to load x4 model as initialization
# save_model = torch.load('checkpoint/sr_hdr_x4_epoch_00020000.pth')
# model_dict = sr_hdr.state_dict()
# state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# sr_hdr.load_state_dict(model_dict)

args.checkpoint = 0
last_epoch = args.checkpoint
if args.checkpoint:
    resume(args.checkpoint, args.scale_factor)
    scheduler.last_epoch = last_epoch


for epoch in trange(last_epoch + 1, args.max_epochs + 1):

    scheduler.step()

    for batch, data in enumerate(train_loader):
        batch_t0 = time.time()

        in_imgs = Variable(data[0].type(torch.float32).cuda())
        ref_LR_HDR = Variable(data[1].type(torch.float32).cuda())
        ref_HR_HDR = Variable(data[2].type(torch.float32).cuda())

        solver.zero_grad()

        bp_t0 = time.time()

        out_LR_HDR, out_HR_HDR = sr_hdr(in_imgs)

        bp_t1 = time.time()

        res_LR = tonemap(ref_LR_HDR) - tonemap(out_LR_HDR)
        loss_LR = res_LR.abs().mean()

        res_HR = tonemap(ref_HR_HDR) - tonemap(F.relu(out_HR_HDR))
        loss_HR = res_HR.abs().mean()

        loss_neg = (F.relu(0.0 - out_HR_HDR)).abs().mean()

        loss = 0.1 * loss_LR + loss_HR + 0.01 * loss_neg

        loss.backward()

        solver.step()

        batch_t1 = time.time()

        index = (epoch - 1) * len(train_loader) + batch

        print(
            '[TRAIN]index({}) Epoch[{}]({}/{}); Loss: {:.6f}; Loss_LR: {:.6f}; Loss_HR: {:.6f}; Loss_neg: {:.6f}; '
            'Forwordtime: {:.4f} sec; Batch: {:.4f} sec; lr: {:.6f}'.
                format(index, epoch, batch + 1, len(train_loader), loss.data, loss_LR.data, loss_HR.data, loss_neg.data,
                       bp_t1 - bp_t0, batch_t1 - batch_t0, solver.param_groups[0]["lr"]))

    if epoch % 1000 == 0:
        save(epoch, args.scale_factor)
