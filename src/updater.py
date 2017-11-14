#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


# https://docs.chainer.org/en/v1.16.0/_modules/chainer/training/updater.html
class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_out, t_out, lam1=100, lam2=1):
        # loss = lam1*(F.mean_absolute_error(x_out, t_out))
        loss = F.softmax_cross_entropy(x_out, t_out)
        chainer.report({'loss': loss}, enc)
        return loss

    def loss_dec(self, dec, x_out, t_out, lam1=100, lam2=1):
        # loss = lam1*(F.mean_absolute_error(x_out, t_out))
        loss = F.softmax_cross_entropy(x_out, t_out)
        chainer.report({'loss': loss}, dec)
        return loss


    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')

        enc, dec = self.enc, self.dec
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        w_in = 256
        w_out = 256

        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        t_out = xp.zeros((batchsize, w_out, w_out)).astype("i")
        # print("x_in; {}".format(x_in.shape))

        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        t_out = Variable(t_out)

        z = enc(x_in)
        x_out = dec(z)
        # print("decoder output; {}".format(x_out.shape))

        enc_optimizer.update(self.loss_enc, enc, x_out, t_out)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out)
        x_in.unchain_backward()
        x_out.unchain_backward()
