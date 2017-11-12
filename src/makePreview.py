# -*- coding:utf-8 -*-
from PIL import Image
import codecs
import os
from subprocess import check_call
import cv2

IMAGES_PATH = "../data/testForFCN"
# IMAGES_PATH = "validate/data/resized_source"
# RESULT_PATH = "out"

# # translate from vidoe to images
# print("traslate from video to frame with a frame number")
# check_call(["../../data/programs/./videoToframe", "../VID_0007.mp4"])


# make prediction which pixel is ingredients or not
for file in os.listdir(IMAGES_PATH):
    image_file = "%s/%s" % (IMAGES_PATH, file) # input image

    print("make prediction")
    # check_call(["python3", "predict.py", "-i", image_file])
    check_call(["python3", "predict.py", "-i", image_file, "-w", "weight/trainTest_withEpoch15times_fcn4Data_withWeightCrossEntropyLoss/chainer_fcn_epoch13.weight"])

# # generate preview
# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (256, 256))

# for file in os.listdir(RESULT_PATH):
#     # result_file = "%s/%s" % (RESULT_PATH, file) # input image
#     # video.write(result_file)
#     img = cv2.imread('./out/pred_frame_{0:05d}.png')

#     cv2.namedWindow('window')
#     cv2.imshow('window', img)
#     # img = cv2.resize(img, (640,480))
# video.release()
