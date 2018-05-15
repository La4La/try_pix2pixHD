#!/usr/bin/env python

from chainer.functions import relu, tanh, leaky_relu
from chainer.functions import average_pooling_2d
from chainer import link
from chainer.links import Convolution2D
from chainer.links import Deconvolution2D
from chainer.links import BatchNormalization
from chainer import serializers
import chainer
from chainer import cuda

from sn.sn_linear import SNLinear
from sn.sn_convolution_2d import SNConvolution2D

import cupy as cp
import numpy as np
from chainer import Variable

from instance_normalization.link import InstanceNormalization

class ResBlock(link.Chain):
    def __init__(self, ch, norm, initialW=None): # numbers of channels of flat blocks are fixed
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn1 = norm(ch)
            self.conv2 = Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = norm(ch)
    def __call__(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return h + x


class GlobalGenerator(link.Chain):
    def __init__(self, in_ch=None, out_ch=3, ins_norm=False, input_size=(512, 1024), num_resblock=9):
        super(GlobalGenerator, self).__init__()
        with self.init_scope():
            self.num_resblock = num_resblock
            if ins_norm:
                norm = InstanceNormalization
            else:
                norm = BatchNormalization
            self.flat1 = Convolution2D(in_ch, 64, 7, 1, 3, initialW=None, nobias=True)
            self.flat1_bn = norm(64)
            self.down1 = Convolution2D(64, 128, 3, 2, 1, initialW=None, nobias=True)
            self.down1_bn = norm(128)
            self.down2 = Convolution2D(128, 256, 3, 2, 1, initialW=None, nobias=True)
            self.down2_bn = norm(256)
            self.down3 = Convolution2D(256, 512, 3, 2, 1, initialW=None, nobias=True)
            self.down3_bn = norm(512)
            self.down4 = Convolution2D(512, 1024, 3, 2, 1, initialW=None, nobias=True)
            self.down4_bn = norm(1024)
            for i in range(self.num_resblock):
                self.add_link('res_{}'.format(i), ResBlock(1024, norm=norm))
            self.up0 = Deconvolution2D(1024, 512, 3, 2, 1, initialW=None, nobias=True, outsize=[int(x / 8) for x in input_size])
            self.up0_bn = norm(512)
            self.up1 = Deconvolution2D(512, 256, 3, 2, 1, initialW=None, nobias=True, outsize=[int(x / 4) for x in input_size])
            self.up1_bn = norm(256)
            self.up2 = Deconvolution2D(256, 128, 3, 2, 1, initialW=None, nobias=True, outsize=[int(x / 2) for x in input_size])
            self.up2_bn = norm(128)
            self.up3 = Deconvolution2D(128, 64, 3, 2, 1, initialW=None, nobias=True, outsize=input_size)
            self.up3_bn = norm(64)
            self.flat2 = Convolution2D(64, out_ch, 7, 1, 3, initialW=None, nobias=True)
    def __call__(self, x, get_global_feature=False):
        h = relu(self.flat1_bn(self.flat1(x)))
        h = relu(self.down1_bn(self.down1(h)))
        h = relu(self.down2_bn(self.down2(h)))
        h = relu(self.down3_bn(self.down3(h)))
        h = relu(self.down4_bn(self.down4(h)))
        for i in range(self.num_resblock):
            h = self['res_{}'.format(i)](h)
        h = relu(self.up0_bn(self.up0(h)))
        h = relu(self.up1_bn(self.up1(h)))
        h = relu(self.up2_bn(self.up2(h)))
        h = relu(self.up3_bn(self.up3(h)))
        if get_global_feature:
            return h
        else:
            out = tanh(self.flat2(h))
            return out


class LocalEnhancer(link.Chain):
    def __init__(self, path_glb, in_ch=None, out_ch=3, ins_norm=False, num_resblock=3, input_size=(512, 1024)):
        super(LocalEnhancer, self).__init__()
        with self.init_scope():
            self.global_network = GlobalGenerator(in_ch)
            serializers.load_npz(path_glb, self.global_network)

            self.num_resblock = num_resblock
            if ins_norm:
                norm = InstanceNormalization
            else:
                norm = BatchNormalization
            self.flat1 = Convolution2D(in_ch, 32, 7, 1, 3)
            self.flat1_bn = norm(32)
            self.down1 = Convolution2D(32, 64, 3, 2, 1)
            self.down1_bn = norm(64)
            for i in range(self.num_resblock):
                self.add_link('res_{}'.format(i), ResBlock(64, norm=norm))
            self.up1 = Deconvolution2D(64, 32, 3, 2, 1, outsize=input_size)
            self.up1_bn = norm(32)
            self.flat2 = Convolution2D(32, out_ch, 7, 1, 3)

    def __call__(self, x):
        h = relu(self.flat1_bn(self.flat1(x)))
        h = relu(self.down1_bn(self.down1(h)))

        x_downsampled = average_pooling_2d(x, 3, 2, 1)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                g = self.global_network(x_downsampled)
                h = h + g

        for i in range(self.num_resblock):
            h = self['res_{}'.format(i)](h)
        h = relu(self.up1_bn(self.up1(h)))
        h = tanh(self.flat2(h))
        return h

class Discriminator(link.Chain):
    def __init__(self, in_ch, wscale=0.02, getIntermFeat=True, conv = Convolution2D):
        self.getIntermFeat = getIntermFeat
        w= chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        kw = 4
        pad = int(np.ceil((kw - 1.0) / 2))
        with self.init_scope():
            self.c1 = conv(in_ch, 64, 4, 2, pad, initialW=w)
            self.c2 = conv(64, 128, 4, 2, pad, initialW=w)
            self.c3 = conv(128, 256, 4, 2, pad, initialW=w)
            self.c4 = conv(256, 512, 4, 1, pad, initialW=w)
            self.c5 = conv(512, 1, 4, 1, pad, initialW=w)
            self.bn0 = BatchNormalization(64)
            self.bn1 = BatchNormalization(128)
            self.bn2 = BatchNormalization(256)
            self.bn3 = BatchNormalization(512)
    def __call__(self, x):
        h1 = leaky_relu(self.c1(x))
        h2 = leaky_relu(self.bn1(self.c2(h1)))
        h3 = leaky_relu(self.bn2(self.c3(h2)))
        h4 = leaky_relu(self.bn3(self.c4(h3)))
        h5 = leaky_relu(self.c5(h4))
        if self.getIntermFeat:
            result = [h1, h2, h3, h4, h5]
        else:
            result = [h5]
        return result

class MultiscaleDiscriminator(link.Chain):
    def __init__(self, in_ch=None, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        with self.init_scope():
            self.D1 = Discriminator(in_ch, getIntermFeat=getIntermFeat)
            self.D2 = Discriminator(in_ch, getIntermFeat=getIntermFeat)
            self.D3 = Discriminator(in_ch, getIntermFeat=getIntermFeat)
    def __call__(self, x):
        h1 = self.D1(x)
        x = average_pooling_2d(x, 3, 2, 1)
        h2 = self.D2(x)
        x = average_pooling_2d(x, 3, 2, 1)
        h3 = self.D3(x)
        result = [h1, h2, h3]
        return result

class Encoder(link.Chain):
    def __init__(self, in_ch=3, out_ch=3):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.flat1 = Convolution2D(in_ch, 32, 7, 1, 3)
            self.flat1_bn = BatchNormalization(32)
            self.down1 = Convolution2D(32, 64, 3, 2, 1)
            self.down1_bn = BatchNormalization(64)
            self.down2 = Convolution2D(64, 128, 3, 2, 1)
            self.down2_bn = BatchNormalization(128)
            self.down3 = Convolution2D(128, 256, 3, 2, 1)
            self.down3_bn = BatchNormalization(256)
            self.down4 = Convolution2D(256, 512, 3, 2, 1)
            self.down4_bn = BatchNormalization(512)
            self.up1 = Deconvolution2D(512, 256, 4, 2, 1)
            self.up1_bn = BatchNormalization(256)
            self.up2 = Deconvolution2D(256, 128, 4, 2, 1)
            self.up2_bn = BatchNormalization(128)
            self.up3 = Deconvolution2D(128, 64, 4, 2, 1)
            self.up3_bn = BatchNormalization(64)
            self.up4 = Deconvolution2D(64, 32, 4, 2, 1)
            self.up4_bn = BatchNormalization(32)
            self.flat2 = Deconvolution2D(32, out_ch, 7, 1, 3)
        self.out_ch = out_ch

    def unique(self, xp, arr):
        arr = arr.flatten()
        arr.sort()
        flags = xp.concatenate((xp.array([True]), arr[1:] != arr[:-1]))
        return arr[flags]

    def __call__(self, x, inst):
        h = relu(self.flat1_bn(self.flat1(x)))
        h = relu(self.down1_bn(self.down1(h)))
        h = relu(self.down2_bn(self.down2(h)))
        h = relu(self.down3_bn(self.down3(h)))
        h = relu(self.down4_bn(self.down4(h)))
        h = relu(self.up1_bn(self.up1(h)))
        h = relu(self.up2_bn(self.up2(h)))
        h = relu(self.up3_bn(self.up3(h)))
        h = relu(self.up4_bn(self.up4(h)))
        # outputs = tanh(self.flat2(h))
        outputs = self.flat2(h)

        # instance-wise average pooling
        outputs_mean = cp.copy(outputs.data)
        inst = inst.data.astype(int)
        ids = np.unique(inst)
        for id in ids:
            indices = list((inst == id).nonzero())
            for j in range(self.out_ch):
                outputs_ins = outputs_mean[indices[0], j, indices[2], indices[3]]
                mean_feat = cp.mean(outputs_ins)
                mean_feat = cp.ones(outputs_ins.shape) * mean_feat
                outputs_mean[indices[0], j, indices[2], indices[3]] = mean_feat

        return Variable(outputs_mean)
