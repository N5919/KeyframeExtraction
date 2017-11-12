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
            # conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=100), # padding = 100
            # conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1),

            # conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1),
            # conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            
            # conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
            # conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            # conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            
            # conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
            # conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            # conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            # conv5_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            # conv5_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            # conv5_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            # fc6 = L.Convolution2D(512, 4096, 7, stride=1, pad=0),
            # fc7 = L.Convolution2D(4096, 4096, 1, stride=1, pad=0),

            # score_fr = L.Convolution2D(4096, n_class, 1, stride=1, pad=0),

            # upscore2 = L.Deconvolution2D(n_class, n_class, ksize=4, stride=2, pad=0, nobias=True),
            # upscore8 = L.Deconvolution2D(n_class, n_class, ksize=16, stride=8, pad=0, nobias=True),
            
            # score_pool3 = L.Convolution2D(256, n_class, 1, stride=1, pad=0),
            # score_pool4 = L.Convolution2D(512, n_class, 1, stride=1, pad=0),
            # upscore_pool4 = L.Deconvolution2D(n_class, n_class, ksize=4, stride=2, pad=0, nobias=True),
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

        # #conv1
        # h = F.relu(self.conv1_2(F.relu(self.conv1_1(x)))) # conv1_1, conv1_2
        # h = F.max_pooling_2d(h, 2, stride=2, pad=0) # pooling
        # pool1 = h # 1/2

        # #conv2
        # h = F.relu(self.conv2_2(F.relu(self.conv2_1(pool1)))) # conv2_1, conv2_2
        # h = F.max_pooling_2d(h, 2, stride=2, pad=0) # pooling
        # pool2 = h # 1/4

        # #conv3
        # h = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(pool2)))))) # conv3_1, conv3_2, conv3_3
        # h = F.max_pooling_2d(h, 2, stride=2, pad=0) # pooling
        # pool3 = h # 1/8

        # #conv4
        # h = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(pool3)))))) # conv4_1, conv4_2, conv4_3
        # h = F.max_pooling_2d(h, 2, stride=2, pad=0) # pooling
        # pool4 = h # 1/16

        # #conv5
        # h = F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(pool4)))))) # conv5_1, conv5_2, conv5_3
        # h = F.max_pooling_2d(h, 2, stride=2, pad=0) # pooling
        # pool5 = h # 1/32

        # # fc6
        # h = F.relu(self.fc6(pool5))
        # h = F.dropout(h, ratio=.5)
        # fc6 = h  # 1/32
        
        # # fc7
        # h = F.relu(self.fc7(fc6))
        # h = F.dropout(h, ratio=.5)
        # fc7 = h  # 1/32
        
        # # score_fr(convolution: tonality of the dimension)
        # h = self.score_fr(fc7)
        # score_fr = h  # 1/32

        # # score_pool3(convolution: tonality of the dimension)
        # h = self.score_pool3(pool3)
        # score_pool3 = h  # 1/8
        
        # # score_pool4(convolution: tonality of the dimension)
        # h = self.score_pool4(pool4)
        # score_pool4 = h  # 1/16
        
        # # upscore2(deconvolution for pool5)
        # h = self.upscore2(score_fr)
        # upscore2 = h  # 1/16

        # # score_pool4c. I do not know why
        # h = score_pool4[:, :,
        #                 5:5 + upscore2.data.shape[2],
        #                 5:5 + upscore2.data.shape[3]]
        # score_pool4c = h  # 1/16

        # # fuse_pool4
        # h = upscore2 + score_pool4c
        # fuse_pool4 = h  # 1/16

        # # upscore_pool4(deconcolution for fuse pool4(score_pool4c) & pool5(upscore2))
        # h = self.upscore_pool4(fuse_pool4)
        # upscore_pool4 = h  # 1/8

        # # score_pool3c. I do not know why
        # h = score_pool3[:, :,
        #                 9:9 + upscore_pool4.data.shape[2],
        #                 9:9 + upscore_pool4.data.shape[3]]
        # score_pool3c = h  # 1/8

        # # fuse_pool3
        # h = upscore_pool4 + score_pool3c
        # fuse_pool3 = h  # 1/8

        # # upscore8(deconcolution for fuse pool3(score_pool3c) & pool4+pool5(upscore_pool4))
        # h = self.upscore8(fuse_pool3)
        # upscore8 = h  # 1/1

        # # score, I do not know why
        # h = upscore8[:, :, 31:31 + x.data.shape[2], 31:31 + x.data.shape[3]]
        # o = h  # 1/1
        
        return o

    ##################
    ###  IoU Loss ####
    ##################

    # def IoU_Loss(self, h, t):
    #     size = 256
    #     union = np.zeros(1, dtype=np.float32)
    #     inter = np.zeros(1, dtype=np.float32)
    #     # # print("softmax.shape: {}, Ground Truth.shape: {}".format(p.shape, t.shape))
    #     # # print("softmax: {}".format(p[0][0][0][0].data))
    #     # # print("Ground Truth: {}".format(t[0][0][0].data)
    #     inter += h[0, :].data*t[:].data
    #     union += ((h[].data*t[:].data)-(h[:, 1, :, :].data*t[:].data))
                
    #     # iou = inter/union
    #     loss = np.ndarray([1], dtype = np.float32)
    #     loss[0] = 1
    #     loss = loss - inter/union
    #     loss = Variable(loss)
    #     print("Loss: {}".format(loss.data))
    #     # loss_val = Variable(loss)
    #     # # print("Loss: {}".format(loss_val.data))
    #     return loss

    ##################
    ## train & test ##
    ##################

    # loss function: Cross Entropy Loss
    
    # def __call__(self, x, t=None, train=False, test=False):
    #     h = self.calc(x, test)
    #     # print("score.shape: {}".format(h.shape))
    #     cw = np.array([2, 1]).astype(np.float32)

    #     if train:
    #         # loss = F.softmax_cross_entropy(h, t, normalize=False) # h = score
    #         loss = F.softmax_cross_entropy(h, t, class_weight = cw)
    #         accuracy = F.accuracy(h, t)
    #         return loss, accuracy
    #     else:
    #         # pred = F.softmax(h)
    #         pred = F.softmax_cross_entropy(h, t)
    #         accuracy = F.accuracy(h, t)
    #         return pred, accuracy

    # loss function: IoU Loss(Intersention over Union Loss)

    # def __call__(self, x, t=None, ft=None, train=False, test=False):
    #     h = self.calc(x, test)
    #     # print("score.shape: {}".format(h.shape))
        
    #     if train:
    #         # loss = F.softmax_cross_entropy(h, t, normalize=False) # h = score
    #         loss = self.IoU_Loss(F.softmax(h), ft)
    #         accuracy = F.accuracy(h, t)
    #         return loss, accuracy
    #     else:
    #         # pred = F.softmax(h)
    #         pred = F.softmax_cross_entropy(h, t)
    #         accuracy = F.accuracy(h, t)
    #         return pred, accuracy

    
    ##################
    ##    PREDICT   ##
    ##################
    
    def __call__(self, x, t=None, train=False, test=False):
        h = self.calc(x, test)
        # print("score.shape: {}".format(h.shape))
        
        if train:
            # loss = F.softmax_cross_entropy(h, t, normalize=False) # h = score
            loss = F.softmax_cross_entropy(h, t)
            accuracy = F.accuracy(h, t)
            return loss, accuracy
        else:
            pred = F.softmax(h)
            # pred = F.softmax_cross_entropy(h, t)
          
            return pred
    
