#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000
# python3 train.py --snapshot_interval 1 -e 1 --display_interval 1

from __future__ import print_function
import sys
import os
import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Encoder
from net import Decoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import out_image

def main():
    parser = argparse.ArgumentParser(description='Chainer U-Net')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')

    # parser.add_argument('--image_path', '-tr', default='dataset', type=str)
    # parser.add_argument('--flowX_dataset', '-tfx', default='dataset', type=str)
    # parser.add_argument('--flowY_dataset', '-tfy', default='dataset', type=str)
    # parser.add_argument('--label_path', '-ta', default='dataset', type=str)
    parser.add_argument('--dataset_path', '-dp', default='../data/For_FCN4/augumentation', type=str)
    parser.add_argument('--train_txt', '-train', default='../data/For_FCN4/augumentation/train.txt', type=str)
    parser.add_argument('--test_txt', '-test', default='../data/For_FCN4/augumentation/val.txt', type=str)

    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='batch size (default value is 1)')
    # parser.add_argument('--initmodel', '-i', default=None, type=str,
    #                     help='initialize the model from given file')
    parser.add_argument('--epoch', '-e', default=200, type=int)
    parser.add_argument('--lr', '-l', default=1e-3, type=float)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    # n_epoch = args.epoch
    # n_class = args.classes
    # batchsize = args.batchsize
    # image_size = args.image_size
    # train_dataset = args.train_dataset
    # flowX_dataset = args.flowX_dataset
    # flowY_dataset = args.flowY_dataset
    # target_dataset = args.label_dataset
    # train_txt = args.train_txt
    # test_txt = args.test_txt

    # Get input image data & label file names
    with open(args.train_txt, "r") as f:
        ls = f.readlines()
    train_names = [l.rstrip('\n') for l in ls]

    with open(args.test_txt, "r") as ft:
        lst = ft.readlines()
    test_names = [l.rstrip('\n') for l in lst]


    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=2)
    # dis = Discriminator(in_ch=12, out_ch=3)

    # if args.gpu >= 0:
    #     chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    #     enc.to_gpu()  # Copy the model to the GPU
    #     dec.to_gpu()
        # dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=args.lr, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    # opt_dis = make_optimizer(dis)

    # loading dataDir
    train_d = FacadeDataset(train_names, args.dataset_path, data_range=(1,5))
    test_d = FacadeDataset(test_names, args.dataset_path, data_range=(1,5))

    # trian & test iteration
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    # trainer.extend(extensions.dump_graph('dec/loss'))
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'enc/loss', 'dec/loss',]), trigger=snapshot_interval)
    trainer.extend(extensions.PlotReport(['epoch', 'iteration', 'enc/loss', 'dec/loss',]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_image(updater, enc, dec, 5, 5, args.seed, args.out), trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # Save the training model
    weight_path = args.out + "/weight"
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    chainer.serializers.save_npz(os.path.join(weight_path, 'enc'), enc)
    chainer.serializers.save_npz(os.path.join(weight_path, 'dec'), dec)
    chainer.serializers.save_npz(os.path.join(weight_path, 'opt_enc'), opt_enc)
    chainer.serializers.save_npz(os.path.join(weight_path, 'opt_dec'), opt_dec)

if __name__ == '__main__':
    main()
