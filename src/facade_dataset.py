

import os

import numpy
from PIL import Image
import six

from io import BytesIO
import os
import pickle
import json
import numpy as np
#
import skimage.io as io
#
from chainer.dataset import dataset_mixin

# download dataset from "../data/For_fcn4"
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, names, dataDir='../data/For_FCN4/augumentation', data_range=(1,300)):
        print("load dataset start")
        print("    from: %s"%dataDir)
        print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0],data_range[1]):
            # img = Image.open(dataDir+"/image/cmp_b%04d.jpg"%i)
            # label = Image.open(dataDir+"/gt/cmp_b%04d.png"%i)
            name = names[i]
            img = Image.open(dataDir+"/image/"+name+".jpg")
            label = Image.open(dataDir+"/gt/"+name+".png")
            w,h = img.size
            r = 286/min(w,h)
            # resize images so that min(w, h) == 286
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label = label.resize((int(r*w), int(r*h)), Image.NEAREST)
            img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
            # label_ = np.asarray(label)-1  # [0, 2)
            label_ = np.asarray(label).astype("i")  # [0, 2)
            label_ = np.reshape(label_[:,:,0], (286,286))
            label = np.zeros((1, img.shape[1], img.shape[2])).astype("i")

            for k in range(img.shape[1]):
                for s in range(img.shape[2]):
                    if label_[k,s] == 0:
                        label[0,k,s] = 0
                    else:
                        label[0,k,s] = 1

            # for j in range(2):
            #     label[j,:] = label_==j
            self.dataset.append((label,img))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        return self.dataset[i][1][:,y_l:y_r,x_l:x_r], self.dataset[i][0][:,y_l:y_r,x_l:x_r]
