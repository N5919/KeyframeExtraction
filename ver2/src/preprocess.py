import numpy as np
from PIL import Image
import cv2

def binarylab(labels, size, nb_class):
    y = np.zeros((nb_class,size,size))
    for i in range(size):
        for j in range(size):
            y[labels[i][j], i, j] = 1
    return y

def load_data(path, crop=True, size=None, mode="label", xp=np):
    img = Image.open(path)
    # img = cv2.imread(path,1)
    #print(img.shape)
    # img = cv2.resize(img,(256,256))
    #print(img.shape)
    
    
    if crop:
        w,h = img.size
        if w < h:
            if w < size:
                img = img.resize((size, size*h//w))
                w, h = img.size
        else:
            if h < size:
                img = img.resize((size*w//h, size))
                w, h = img.size
        img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))

    if mode=="label":
        y = xp.asarray(img, dtype=xp.int32)
        labels = xp.zeros((size, size), dtype=xp.int32)
        mask = y == 255
        #mask = mask.astype(xp.int32)
        y[mask] = -1
        for i in range(size):
            for j in range(size):
                if y[i][j][0] == 0 and y[i][j][1] == 0 and y[i][j][2] == 0:
                    labels[i][j] = 0
                else:
                    labels[i][j] = 1

        # for i in range(size):
        #     for j in range(size):
        #         labels[i][j] = y[i][j]
        # return y
        return labels

    elif mode=="data":
        x = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        return x

    elif mode=="predict":
        return img
