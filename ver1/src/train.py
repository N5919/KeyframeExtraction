

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions
from create_chainer_model import copy_model
from chainer.links.caffe import CaffeFunction

import sys
import os
import argparse

from model import FCN
from preprocess import load_data

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Chainer Fully Convolutional Network')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--train_dataset', '-tr', default='dataset', type=str)
parser.add_argument('--flowX_dataset', '-tfx', default='dataset', type=str)
parser.add_argument('--flowY_dataset', '-tfy', default='dataset', type=str)
parser.add_argument('--target_dataset', '-ta', default='dataset', type=str)
parser.add_argument('--train_txt', '-tt', default='train_txt', type=str)
parser.add_argument('--val_txt', '-tv', default='val_txt', type=str) # val_txt
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--classes', default=2, type=int)
args = parser.parse_args()

n_epoch = args.epoch
n_class = args.classes
batchsize = args.batchsize
image_size = args.image_size
train_dataset = args.train_dataset
flowX_dataset = args.flowX_dataset
flowY_dataset = args.flowY_dataset
target_dataset = args.target_dataset
train_txt = args.train_txt
val_txt = args.val_txt

def validation(model):
    sum_accuracy = 0
    sum_loss = 0

    ##################
    ## loading Data ##
    ##################
    with open(val_txt, "r") as fval:
        lsVal = fval.readlines()
        valNames = [l.rstrip('\n') for l in lsVal]
        val_data = len(valNames)

    # val_data = 1
    for k in range(val_data):
        xVal = xp.zeros((1, 3, image_size, image_size), dtype=np.float32)
        yVal = xp.zeros((1, image_size, image_size), dtype=np.int32)
        for i in range(0, 1):
            valName = valNames[i]
            xValpath = train_dataset+valName+".jpg"
            yValpath = target_dataset+valName+".png"
            xVal[i] = load_data(xValpath, crop=True, size=256, mode="data", xp=xp)
            yVal[i] = load_data(yValpath, crop=True, size=256, mode="label", xp=xp)
            
        xVal = Variable(xVal)
        yVal = Variable(yVal)
        
        ########################
        ## Cross Entropy Loss ##
        ########################       
        loss, accuracy = model(xVal, yVal, train=False)
        sum_loss += loss.data
        sum_accuracy += accuracy.data
        sys.stdout.write("\r%s" % "Test: batch: {}/{}, loss: {}, accuracy: {}".format(k+1, val_data, loss.data, accuracy.data))

        
    return sum_loss/val_data, sum_accuracy/val_data
    
with open(train_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
n_data = len(names)
n_iter = n_data // batchsize
gpu_flag = True if args.gpu > 0 else False

model = FCN(n_class)

if args.initmodel:
    print('load Pixel Objectness caffemodel')
    ref = CaffeFunction('pixel_objectness.caffemodel')
    model = FCN()
    print('copy weights')
    copy_model(ref, model)
    # serializers.load_npz(args.initmodel, model)
    print("Load initial weight")

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer parameters.
optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_fcn')



####################
#### training ######
####################

print("## INFORMATION ##")
print("Num Data: {}, Batchsize: {}, Iteration {}".format(n_data, batchsize, n_iter))

result = []
result.append([0, 1, 0, 1, 0]) 

print("-"*40)
for epoch in range(1, n_epoch+1):
    sum_loss = 0
    sum_accuracy = 0
    print('epoch', epoch)
    for i in range(n_iter):

        model.zerograds()
        indices = range(i * batchsize, (i+1) * batchsize)

        x = xp.zeros((batchsize, 3, image_size, image_size), dtype=np.float32)
        y = xp.zeros((batchsize, image_size, image_size), dtype=np.int32)
        for j in range(batchsize):
            name = names[i*batchsize + j]
            xpath = train_dataset+name+".jpg"
            ypath = target_dataset+name+".png"
            x[j] = load_data(xpath, crop=True, size=256, mode="data", xp=xp)
            y[j] = load_data(ypath, crop=True, size=256, mode="label", xp=xp)

        
        x = Variable(x)
        y = Variable(y)
        loss, accuracy = model(x, y, train=True)                 
        sum_loss += loss.data
        sum_accuracy += accuracy.data
        sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}, accuracy: {}".format(i+1, n_iter, loss.data, accuracy.data))
        sys.stdout.flush()
        

        loss.backward()
        optimizer.update()

        
    ####################
    ### validation #####
    ####################
    testLoss, testAccuracy = validation(model)


    
    sys.stdout.write("\r%s" % "epoch: {}/{}, loss: {}, accuracy: {}, test loss: {}, test accuracy: {}".format(epoch, n_epoch, sum_loss/n_iter, sum_accuracy/n_iter, testLoss, testAccuracy))
    result.append([epoch, sum_loss/n_iter, sum_accuracy/n_iter, testLoss, testAccuracy])
    
    print("\n"+"-"*40)
    serializers.save_npz('weight/trainTest_withEpoch15times_fcn4Data_withWeightCrossEntropyLoss/chainer_fcn_epoch'+str(epoch)+'.weight', model)
    serializers.save_npz('weight/trainTest_withEpoch15times_fcn4Data_withWeightCrossEntropyLoss/chainer_fcn_epoch'+str(epoch)+'.state', optimizer)
    print('save weight')
    
if not os.path.exists("weight"):
    os.mkdir("weight")
serializers.save_npz('weight/chainer_fcn.weight', model)
serializers.save_npz('weight/chainer_fcn.state', optimizer)
print('save weight')


###################################
### Create a Graph for logging ####
###################################

# train & validation
# graph
x_epoch = [d[0] for d in result]
y_loss = [d[1] for d in result]
y_accuracy = [d[2] for d in result]
y_testLoss = [d[3] for d in result]
y_testAccuracy = [d[4] for d in result]

plt.title("loss & accuracy for each epoch")
plt.xlabel("epoch")
plt.ylabel("loss/accuracy")
plt.ylim([0, 1])
y_loss, = plt.plot(x_epoch,y_loss,color="red")
y_accuracy, = plt.plot(x_epoch,y_accuracy,color="blue")
y_testLoss, = plt.plot(x_epoch,y_testLoss,color="orange")
y_testAccuracy, = plt.plot(x_epoch,y_testAccuracy,color="green")

plt.grid(True)
plt.legend((y_loss, y_accuracy, y_testLoss, y_testAccuracy), ("loss", "accuracy", "test loss", "test accuracy"), loc = 2)
plt.savefig("log.png")
# plt.show()

