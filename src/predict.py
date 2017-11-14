#!/usr/bin/env python3

# python3 predict.py -ip ../segmentation/validate/data/source/frame_03611.jpg

from __future__ import print_function
from PIL import Image
import sys
import os
import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import Variable

from net import Encoder
from net import Decoder
from updater import FacadeUpdater

from facade_dataset import FacadeDataset
from facade_visualizer import out_image

from color_map import make_color_map

def load_data(path, mode="image"):
    img = Image.open(path)
    w,h = img.size
    r = 256/min(w,h)
    # resize images so that min(w, h) == 256
    img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
    # w,h = img.size
    # if w < h:
    #     if w < size:
    #             img = img.resize((size, size*h//w))
    #             w, h = img.size
    #     else:
    #         if h < size:
    #             img = img.resize((size*w//h, size))
    #             w, h = img.size
    #     img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))

    if mode == "image":
        # x = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
        return img

def save_image(x, name, save_dir, mode="pred"):
    if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    out_path = 'result/pred/pred_{}.png'.format(name)
    if mode == "pred":
        _, C, H, W = x.shape
        x = x.reshape((C, H, W))
        x = x.transpose(1, 2, 0)
        Image.fromarray(x).convert('RGB').save(out_path)


def main():
    parser = argparse.ArgumentParser(description='Chainer U-Net: predict')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--mode', '-m', default="image", type=str,
                        help='Mode Setting: image/video')
    parser.add_argument('--image_path', '-ip', default=None, type=str)
    # parser.add_argument('--video_path', '-vp' default=None, type=str)
    parser.add_argument('--enc_weight', '-ew', default="result_facade/enc_iter_4190.npz", type=str)
    parser.add_argument('--dec_weight', '-dw', default="result_facade/dec_iter_4190.npz", type=str)
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--out_path', '-o', default="result/pred", type=str)
    # parser.add_argument('--clop', "-c", default=True, type=bool)
    # parser.add_argument('--clopsize', "-s", default=256, type=int)
    args = parser.parse_args()

    # Get input image names
    img_name = args.image_path.split("/")[-1].split(".")[0]
    # color_map = make_color_map()
    # print(color_map.shape)

    # Set up a neural network to prediction
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=2)
    # Load parameters
    serializers.load_npz(args.enc_weight, enc)
    serializers.load_npz(args.dec_weight, dec)

    # Load input image/video
    pred_d = load_data(args.image_path, mode="image")
    pred_d = np.expand_dims(pred_d, axis=0) # (3, 256, 256) -> (1, 3, 256, 256)

    z = enc(pred_d)
    pred_d = dec(z)

    mask = np.zeros((1, 2, 256, 256)).astype("f")
    mask = pred_d.data
    mask = np.asarray(np.clip(mask * 128 + 128, 0.0, 255.0), dtype=np.uint8)
    save_image(mask, img_name, args.out_path)

if __name__ == '__main__':
    main()
