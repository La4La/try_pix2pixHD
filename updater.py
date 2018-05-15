#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable

class GlobalUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        self.D_num = 3
        self.xp = self.gen.xp
        self.size = kwargs.pop('size')
        super(GlobalUpdater, self).__init__(*args, **kwargs)
        self.label_true = []
        self.label_false = []

    def make_labels(self, D):
        for d in D:
            self.label_true.append(self.xp.full(d[-1].shape, 1, dtype=self.xp.float32))
            self.label_false.append(self.xp.full(d[-1].shape, 0, dtype=self.xp.float32))

    def loss_gen(self, gen, output, D_fake, D_real, alpha=10):
        loss_GAN, loss_FM = 0, 0
        if not self.label_true:
            self.make_labels(D_fake)
        for i in range(self.D_num):
            # GAN loss
            loss_GAN += 0.5 * F.mean_squared_error(D_fake[i][-1], self.label_true[i])
            # Feature Matching loss
            for j in range(len(D_fake) - 1):
                loss_FM += alpha * F.mean_absolute_error(D_fake[i][j], D_real[i][j]) #/ self.D_num
        chainer.report({'loss_GAN': loss_GAN, 'loss_FM': loss_FM}, gen)
        return loss_GAN + loss_FM

    def loss_dis(self, dis, D_fake, D_real):
        loss = 0
        for i in range(self.D_num):
            loss_fake = 0.5 * F.mean_squared_error(D_fake[i][-1], self.label_false[i])
            loss_real = 0.5 * F.mean_squared_error(D_real[i][-1], self.label_true[i])
            loss += (loss_fake + loss_real)
        chainer.report({'loss_GAN': loss}, dis)
        return loss


    def update_core(self):
        self._iter += 1

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        label = []
        truth = []
        for i in range(batchsize):
            label.append(np.asarray(batch[i][0]).astype("f"))
            truth.append(np.asarray(batch[i][1]).astype("f"))
        label = Variable(self.xp.asarray(label))
        truth = Variable(self.xp.asarray(truth))

        output = self.gen(label)

        fake_input = F.concat((output, label), axis=1)
        real_input = F.concat((truth, label), axis=1)
        D_fake = self.dis(fake_input)
        D_real = self.dis(real_input)

        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen_optimizer.update(self.loss_gen, self.gen, output, D_fake, D_real)
        output.unchain_backward()
        dis_optimizer.update(self.loss_dis, self.dis, D_fake, D_real)

