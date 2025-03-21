from chainer import serializers
import numpy as np
from PIL import Image
import os
import argparse
import cv2

from model import FCN
from preprocess import load_data
from color_map import make_color_map

parser = argparse.ArgumentParser(description='Chainer Fully Convolutional Network: predict')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--image_path', '-i', default=None, type=str)
parser.add_argument('--weight', '-w', default="weight/chainer_fcn.weight", type=str)
# parser.add_argument('--classes', default=21, type=int)
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--clop', "-c", default=True, type=bool)
parser.add_argument('--clopsize', "-s", default=256, type=int)
args = parser.parse_args()

img_name = args.image_path.split("/")[-1].split(".")[0]

color_map = make_color_map()
model = FCN(n_class=args.classes)
# serializers.load_npz('weight/chainer_fcn.weight', model)
serializers.load_npz(args.weight, model)

o = load_data(args.image_path, crop=args.clop, size=args.clopsize, mode="predict")
x = load_data(args.image_path, crop=args.clop, size=args.clopsize, mode="data")
x = np.expand_dims(x, axis=0)
pred = model(x).data
print(pred.shape)
pred = pred[0].argmax(axis=0)

row, col = pred.shape
dst = np.ones((row, col, 3))
# for i in range(21):
for i in range(2):
    dst[pred == i] = color_map[i]
img = Image.fromarray(np.uint8(dst))

b,g,r = img.split()
img = Image.merge("RGB", (r, g, b))

# img.show()
# img.save("out/{}.png".format(img_name))
# img.save("validate/data/mask/{}.png".format(img_name)) # validate
# img.save("makeVideo/mask/{}.png".format(img_name)) # make video

trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
w, h = img.size
for x in range(w):
    for y in range(h):
        pixel = img.getpixel((x, y))
        if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0)or \
           (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
            continue
        trans.putpixel((x, y), pixel)
#o.paste(trans, (0,0), trans)




###############
## validate ###
###############

# if not os.path.exists("out"):
#     os.mkdir("out")
# o.save("out/original.jpg")
# trans.save("out/pred.png")

# o = cv2.imread("out/original.jpg", 1)
# p = cv2.imread("out/pred.png", 1)

# pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

# cv2.imwrite("out/pred_{}.png".format(img_name), pred)

# os.remove("out/original.jpg")
# os.remove("out/pred.png")

# if not os.path.exists("out"):
#     os.mkdir("out")
# o.save("out/original.jpg")
# trans.save("out/pred.png")

# o = cv2.imread("out/original.jpg", 1)
# p = cv2.imread("out/pred.png", 1)

# pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

# cv2.imwrite("validate/data/mask/pred_{}.png".format(img_name), pred)

# os.remove("out/original.jpg")
# os.remove("out/pred.png")



# #################
# # make video ####
# #################


if not os.path.exists("makeVideo"):
    os.mkdir("makeVideo")
o.save("makeVideo/original.jpg")
trans.save("makeVideo/pred.png")

o = cv2.imread("makeVideo/original.jpg", 1)
p = cv2.imread("makeVideo/pred.png", 1)

pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

cv2.imwrite("makeVideo/pred_{}.png".format(img_name), pred)

os.remove("makeVideo/original.jpg")
os.remove("makeVideo/pred.png")
