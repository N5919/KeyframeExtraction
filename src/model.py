import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable
from chainer import initializers

import chainercv
import chainer.links as L

class FCN(chainer.Chain):
    def __init__(self, n_class=1):
        super(FCN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            pool3=L.Convolution2D(256, n_class, 1, stride=1, pad=0),
            pool4=L.Convolution2D(512, n_class, 1, stride=1, pad=0),
            pool5=L.Convolution2D(512, n_class, 1, stride=1, pad=0),

            upsample4=L.Deconvolution2D(n_class, n_class, ksize= 4, stride=2, pad=1),
            upsample5=L.Deconvolution2D(n_class, n_class, ksize= 8, stride=4, pad=2),
            upsample =L.Deconvolution2D(n_class, n_class, ksize=16, stride=8, pad=4),
        )
        self.train = False

    def calc(self, x, test=False):
        h = F.relu(self.conv1_2(F.relu(self.conv1_1(x)))) # conv1_1, conv1_2
        h = F.max_pooling_2d(h, 2, stride=2) # pooling
        h = F.relu(self.conv2_2(F.relu(self.conv2_1(h)))) # conv2_1, conv2_2
        h = F.max_pooling_2d(h, 2, stride=2) # pooling
        h = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h)))))) # conv3_1, conv3_2, conv3_3
        p3 = F.max_pooling_2d(h, 2, stride=2) # pooling
        h = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(p3)))))) # conv4_1, conv4_2, conv4_3
        p4 = F.max_pooling_2d(h, 2, stride=2) # pooling
        h = F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(p4)))))) # conv5_1, conv5_2, conv5_3
        p5 = F.max_pooling_2d(h, 2, stride=2) # pooling


        p3 = self.pool3(p3)
        p4 = self.upsample4(self.pool4(p4)) # convolution with pool4's result + deconvolution
        p5 = self.upsample5(self.pool5(p5)) # convolution with pool5's result + deconvolution

        h = p3 + p4 + p5
        o = self.upsample(h)


    ##################
    ## train & test ##
    ##################

    # loss function: Cross Entropy Loss

    def __call__(self, x, t=None, train=False, test=False):
        h = self.calc(x, test)
        # print("score.shape: {}".format(h.shape))

        # change output array shape since using sigmoid function as activate function
        h2 = np.reshape(h.data, (1, 256, 256))
        h2 = Variable(h2)

        if train:
            loss = F.sigmoid_cross_entropy(h2, t)
            accuracy = F.accuracy(h, t)
            return loss, accuracy
        else:
            # pred = F.softmax(h)
            pred = F.softmax_cross_entropy(h, t)
            accuracy = F.accuracy(h, t)
            return pred, accuracy



    ##################
    ##    PREDICT   ##
    ##################

    # def __call__(self, x, t=None, train=False, test=False):
    #     h = self.calc(x, test)
    #     # print("score.shape: {}".format(h.shape))
    #
    #     if train:
    #         # loss = F.softmax_cross_entropy(h, t, normalize=False) # h = score
    #         loss = F.softmax_cross_entropy(h, t)
    #         accuracy = F.accuracy(h, t)
    #         return loss, accuracy
    #     else:
    #         pred = F.softmax(h)
    #         # pred = F.softmax_cross_entropy(h, t)
    #
    #         return pred
