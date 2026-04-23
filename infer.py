#python infer.py --evaluate checkpoint_58.4/ckpt_best.pth.tar -cfg checkpoint_58.4/w32_adam_lr1e-3.yaml

from __future__ import print_function, absolute_import, division

import random
import os
import time
import datetime
import argparse
import numpy as np
import shutil
import json
import os.path as path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from lib.config import cfg

from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe, sym_penalty
from common.camera import get_uvd2xyz
from utils.prepare_data_h3wb import Human3WBDataset, TRAIN_SUBJECTS, TEST_SUBJECTS

from models.graph_sh import GraphSH
import models.graph_hrnet_multi_branch_58 as ghrmb_58
import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h3wb_', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-cfg', '--configuration', default='w32_adam_lr1e-3.yaml', type=str, metavar='NAME',
                        help='configuration file')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 20)')

    # Model arguments
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('-z', '--hid_dim', default=64, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=40, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--lamda', '--weight_L1_norm', default=0, type=float, metavar='N', help='scale of L1 Norm')
    parser.add_argument('--num_workers', default=24, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-2, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=500, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--post_refine', dest='post_refine', action='store_true', help='if use post-refine layers')
    parser.set_defaults(post_refine=False)
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    parser.add_argument('--gcn', default='dc_preagg', type=str, metavar='NAME', help='type of gcn')
    parser.add_argument('-n', '--name', default='', type=str, metavar='NAME', help='name of model')

    # Experimental
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args
def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', args.dataset + 'train.npz')
    dataset_test_path = path.join('data', args.dataset + 'test.npz')
    dataset = Human3WBDataset(dataset_path, dataset_test_path)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(dataset)

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)



    cudnn.benchmark = True
    device = torch.device("cuda:0")

    # Create model
    print("==> Creating model...")

    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton()).to(device)

    cfg.merge_from_file(args.configuration)

    model_pos = ghrmb_58.get_pose_net(cfg, True, adj, p_dropout, args.gcn, dataset.skeleton().joints_group()).to(device)

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)
    # Optionally resume from a checkpoint
    if args.evaluate:
        ckpt_path = args.evaluate

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint")
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

    print('==> Infer...')

    torch.set_grad_enabled(False)
    model_pos.eval()

    input_2d=torch.tensor(keypoints['S1']['Posing'][3])


    input_2d = input_2d.to(device)

    body_3d, face_3d, left_hand_3d, right_hand_3d = model_pos(input_2d)
    output = torch.cat((body_3d, face_3d, left_hand_3d, right_hand_3d), dim=1)
    #output = output - (output[:, 11:12, :] + output[:, 12:13, :]) / 2
    torch.save(output, "3d_keypoints.pt")
    print("results saved to 3d_keypoints.pt")
    return 0


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main(parse_args())
