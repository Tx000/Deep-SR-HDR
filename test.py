import os
from utils import *
import torch
from torch.autograd import Variable
import argparse
import cv2
from model import Deep_SR_HDR

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
        
        out_LR_HDR, out_HDR = sr_hdr(inputs)
        out_HDR = torch.clamp(out_HDR, 0, 1)
        out_HDR_tone = tonemap(out_HDR) * 255

    out_HDR = np.transpose(np.squeeze(out_HDR.data.cpu().numpy()), axes=(2, 1, 0))
    out_HDR_tone = np.transpose(np.squeeze(out_HDR_tone.data.cpu().numpy()), axes=(2, 1, 0))

    if args.use_cuda:
        torch.cuda.empty_cache()

    imsave(out_HDR, os.path.join(args.results_dir, 'test_{:03d}_{:03d}_HDR.hdr'.format(idx,idx + 1)))
    cv2.imwrite(os.path.join(args.results_dir, 'test_{:03d}_{:03d}_tonemapped.png'.format(idx,idx + 1)), out_HDR_tone[:, :, ::-1])

