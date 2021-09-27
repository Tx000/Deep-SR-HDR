import os
from utils import *
import torch
from torch.autograd import Variable
import argparse
import time
import cv2
from model import Deep_SR_HDR
import h5py

torch.cuda.set_device(0)

test_check = 9000
scale_factor = 2

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', default=True, type=bool, help='use cuda or not')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/sr_hdr_x{}_epoch_{:08d}.pth'.format(scale_factor, test_check))
parser.add_argument('--results_dir', type=str, default='results')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--c_dim', type=int, default=3)
parser.add_argument('--dataset', type=str, default='./dataset_full/Test')
parser.add_argument('--test_h', type=int, default=960)
parser.add_argument('--test_w', type=int, default=1440)
parser.add_argument('--scale_factor', type=int, default=scale_factor, help='image height')

parser.add_argument('--nReslayer', type=int, default=3, help='nReslayer of RB')
parser.add_argument('--nDenselayer', type=int, default=4, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--filters_in', type=int, default=6, help='number of channels in')
parser.add_argument('--filters_out', type=int, default=3, help='number of channels out')
parser.add_argument('--groups', type=int, default=8, help='number of groups')
args = parser.parse_args()

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

MU = 5000. # tunemapping parameter
def tonemap_np(images):  # input/output 0~1
    return np.log(1 + MU * images) / np.log(1 + MU)


def tonemap(images):  # input/output 0~1
    return torch.log(1.0 + MU * images) / np.log(1.0 + MU)


GAMMA = 2.2 # LDR&HDR domain transform parameter
def LDR2HDR(img, expo): # input/output 0~1
    return img ** GAMMA / expo


def arrenge(img):
    H, W, C = img.shape
    img_ = np.zeros(shape=(W, H, C))
    for i in range(W):
        for j in range(H):
            img_[i, j, :] = img[j, i, :]
    return img_


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    _, _, h, w = x.shape
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return x[:, :, max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)]


def resize(x, scale_factor):
    _, _, h, w = x.shape
    x = np.transpose(np.squeeze(x), axes=(1, 2, 0))
    x = np.clip(cv2.resize(x, dsize=(w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC), 0, 1)
    x = np.expand_dims(np.transpose(x, axes=(2, 0, 1)), axis=0)
    return x


def get_input(scene_dir, image_size, scale_factor):
    hf = h5py.File(scene_dir)
    inputs = np.array(hf.get('IN'))
    inputs = center_crop(inputs, image_size)
    inputs = resize(inputs, scale_factor)
    ref_HDR = np.array(hf.get('GT'))
    ref_HDR = center_crop(ref_HDR, image_size)
    return inputs, ref_HDR


scene_dirs = sorted(os.listdir(args.dataset))
nScenes = len(scene_dirs)
num_batch = int(np.ceil(nScenes/args.batch_size))

sr_hdr = Deep_SR_HDR(args=args)
if args.use_cuda:
    hdr_sr = sr_hdr.cuda()
sr_hdr.load_state_dict(torch.load(args.checkpoint_dir, map_location=torch.device('cpu')))
sr_hdr.eval()

for idx in range(nScenes):
    print('processing no. %d' % (idx + 1), end='\r')

    input_features, ref_HDR = get_input(os.path.join(args.dataset, scene_dirs[idx]),
                                        [args.test_w, args.test_h], scale_factor=args.scale_factor)

    with torch.no_grad():
        inputs = Variable(torch.from_numpy(input_features)).cuda()
        ref = Variable(torch.from_numpy(ref_HDR)).cuda()

    with torch.no_grad():
        out_LR_HDR, out_HDR = sr_hdr(inputs)
        out_HDR = torch.clamp(out_HDR, 0, 1)
        out_HDR_tone = tonemap(out_HDR) * 255

    out_HDR = arrenge(np.transpose(np.squeeze(out_HDR.data.cpu().numpy()), axes=(1, 2, 0)))
    out_HDR_tone = arrenge(np.transpose(np.squeeze(out_HDR_tone.data.cpu().numpy()), axes=(1, 2, 0)))

    if args.use_cuda:
        torch.cuda.empty_cache()

    imsave(out_HDR, os.path.join(args.results_dir, 'test_{:03d}_{:03d}_HDR.hdr'.format(idx,idx + 1)))
    cv2.imwrite(os.path.join(args.results_dir, 'test_{:03d}_{:03d}_tonemapped.png'.format(idx,idx + 1)), out_HDR_tone[:, :, ::-1])

