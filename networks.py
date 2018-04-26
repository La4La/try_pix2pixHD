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
            self.bn1 = BatchNormalization(ch)
            self.conv2 = Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = BatchNormalization(ch)
    def __call__(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return h + x


class GlobalNetwork(link.Chain):
    def __init__(self, in_ch, norm):
        super(GlobalNetwork, self).__init__()
        with self.init_scope():
            self.flat1 = Convolution2D(in_ch, 64, 7, 1, 3, initialW=None, nobias=True)
            self.flat1_bn = norm(64)
            self.down1 = Convolution2D(64, 128, 3, 2, 1, initialW=None, nobias=True)
            self.down1_bn = norm(128)
            self.down2 = Convolution2D(128, 256, 3, 2, 1, initialW=None, nobias=True)
            self.down2_bn = norm(256)
            self.down3 = Convolution2D(256, 512, 3, 2, 1, initialW=None, nobias=True)
            self.down3_bn = norm(512)
            self.res1 = ResBlock(512, norm=norm)
            self.res2 = ResBlock(512, norm=norm)
            self.res3 = ResBlock(512, norm=norm)
            self.res4 = ResBlock(512, norm=norm)
            self.res5 = ResBlock(512, norm=norm)
            self.res6 = ResBlock(512, norm=norm)
            self.res7 = ResBlock(512, norm=norm)
            self.res8 = ResBlock(512, norm=norm)
            self.res9 = ResBlock(512, norm=norm)
            self.up1 = Deconvolution2D(512, 256, 4, 2, 1, initialW=None, nobias=True)
            self.up1_bn = norm(256)
            self.up2 = Deconvolution2D(256, 128, 4, 2, 1, initialW=None, nobias=True)
            self.up2_bn = norm(128)
            self.up3 = Deconvolution2D(128, 64, 4, 2, 1, initialW=None, nobias=True)
            self.up3_bn = norm(64)
    def __call__(self, x):
        h = relu(self.flat1_bn(self.flat1(x)))
        h = relu(self.down1_bn(self.down1(h)))
        h = relu(self.down2_bn(self.down2(h)))
        h = relu(self.down3_bn(self.down3(h)))
        h = relu(self.res1(h))
        h = relu(self.res2(h))
        h = relu(self.res3(h))
        h = relu(self.res3(h))
        h = relu(self.res4(h))
        h = relu(self.res5(h))
        h = relu(self.res6(h))
        h = relu(self.res7(h))
        h = relu(self.res8(h))
        h = relu(self.res9(h))
        h = relu(self.up1_bn(self.up1(h)))
        h = relu(self.up2_bn(self.up2(h)))
        h = relu(self.up3_bn(self.up3(h)))
        return h


class GlobalGenerator(link.Chain):
    def __init__(self, in_ch=36, out_ch=3, ins_norm=False):
        super(GlobalGenerator, self).__init__()
        with self.init_scope():
            if ins_norm:
                norm = InstanceNormalization
            else:
                norm = BatchNormalization
            self.global_network = GlobalNetwork(in_ch, norm=norm)
            self.flat2 = Convolution2D(64, out_ch, 7, 1, 3, initialW=None, nobias=True)
    def __call__(self, x):
        h = self.global_network(x)
        # out = tanh(self.flat2(h))
        out = self.flat2(h)
        return out


class LocalEnhancer(link.Chain):
    def __init__(self, in_ch, out_ch, path_glb, path_loc, FineTune=False):
        super(LocalEnhancer, self).__init__()
        with self.init_scope():
            self.global_network = GlobalNetwork(in_ch)
            serializers.load_npz(path_glb, self.global_network)

            self.flat1 = Convolution2D(in_ch, 32, 7, 1, 3)
            self.flat1_bn = BatchNormalization(32)
            self.down1 = Convolution2D(32, 64, 3, 2, 1)
            self.down1_bn = BatchNormalization(64)
            self.res1 = ResBlock(64)
            self.res2 = ResBlock(64)
            self.res3 = ResBlock(64)
            self.up1 = Deconvolution2D(64, 32, 4, 2, 1)
            self.up1_bn = BatchNormalization(32)
            self.flat2 = Convolution2D(32, out_ch, 7, 1, 3)

            self.FineTune = FineTune

    def __call__(self, x):
        h = relu(self.flat1_bn(self.flat1(x)))
        h = relu(self.down1_bn(self.down1(h)))

        x_downsampled = average_pooling_2d(x, 3, 2, 1)
        if self.FineTune:
            g = self.global_network(x_downsampled)
        else:
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    g = self.global_network(x_downsampled)

        h = relu(self.res1(h + g))
        h = relu(self.res2(h))
        h = relu(self.res3(h))
        h = relu(self.up1_bn(self.up1(h)))
        h = tanh(self.flat2(h))
        return h

class Discriminator(link.Chain):
    def __init__(self, in_ch, wscale=0.02, getIntermFeat=True, SNGAN=True):
        self.getIntermFeat = getIntermFeat
        w= chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        if SNGAN:
            conv = SNConvolution2D
        else:
            conv = Convolution2D
        # notice: receptive field is a little bit smaller than 70*70
        with self.init_scope():
            self.c1 = conv(in_ch, 64, 4, 2, 1, initialW=w)
            self.c2 = conv(64, 128, 4, 2, 1, initialW=w)
            self.c3 = conv(128, 256, 4, 2, 1, initialW=w)
            self.c4 = conv(256, 512, 4, 1, 1, initialW=w)
            self.c5 = conv(512, 1, 4, 1, 1, initialW=w)
    def __call__(self, x):
        h1 = leaky_relu(self.c1(x))
        h2 = leaky_relu(self.c2(h1))
        h3 = leaky_relu(self.c3(h2))
        h4 = leaky_relu(self.c4(h3))
        h5 = leaky_relu(self.c5(h4))
        if self.getIntermFeat:
            result = [h1, h2, h3, h4, h5]
        else:
            result = [h5]
        return result

class MultiscaleDiscriminator(link.Chain):
    def __init__(self, in_ch=39, SNGAN=True):
        super(MultiscaleDiscriminator, self).__init__()
        with self.init_scope():
            self.D1 = Discriminator(in_ch, SNGAN=SNGAN)
            self.D2 = Discriminator(in_ch, SNGAN=SNGAN)
            self.D3 = Discriminator(in_ch, SNGAN=SNGAN)
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
