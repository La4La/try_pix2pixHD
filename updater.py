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


    def loss_gen(self, gen, output, D_fake, D_real, alpha=10):
        loss_GAN, loss_FM = 0, 0

        for i in range(self.D_num):
            # GAN loss
            b, _, h, w = D_fake[i][-1].shape
            loss_GAN_i = F.sum(F.softplus(-D_fake[i][-1]))/(b*h*w)
            chainer.report({'loss_GAN'+str(i): loss_GAN_i}, gen)
            loss_GAN += loss_GAN_i

            # Feature Matching loss
            loss_FM_i = 0
            for j in range(len(D_fake) - 1):
                loss_FM_i += F.mean_absolute_error(D_fake[i][j], D_real[i][j])
            loss_FM_i = alpha * (1.0 / self.D_num) * loss_FM_i
            chainer.report({'loss_FM'+str(i): loss_FM_i}, gen)
            loss_FM += loss_FM_i

        chainer.report({'loss_GAN': loss_GAN, 'loss_FM': loss_FM}, gen)
        return loss_GAN + loss_FM

    def loss_dis(self, dis, D_fake, D_real):
        loss = 0
        for i in range(self.D_num):
            b, _, h, w = D_real[i][-1].shape
            loss_fake = F.sum(F.softplus(D_fake[i][-1]))/(b*h*w)
            loss_real = F.sum(F.softplus(-D_real[i][-1]))/(b*h*w)
            loss_i = loss_fake + loss_real
            chainer.report({'loss'+str(i): loss_i}, dis)
            loss += loss_i
        chainer.report({'loss_GAN': loss}, dis)
        return loss


    def update_core(self):
        self._iter += 1

        # batch = [oneHot_label_map, edge_map, real_image, feature_map]
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        label = []
        truth = []
        for i in range(batchsize):
            label.append(batch[i][0].data)
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



class GlobalUpdater2(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.enc = kwargs.pop('models')
        self._iter = 0
        self.D_num = 3
        self.xp = self.gen.xp
        self.size = kwargs.pop('size')
        super(GlobalUpdater2, self).__init__(*args, **kwargs)


    def loss_gen(self, gen, output, D_fake, D_real, alpha=10):
        loss_GAN, loss_FM = 0, 0

        for i in range(self.D_num):
            # GAN loss
            b, _, h, w = D_fake[i][-1].shape
            loss_GAN_i = F.sum(F.softplus(-D_fake[i][-1]))/(b*h*w)
            chainer.report({'loss_GAN'+str(i): loss_GAN_i}, gen)
            loss_GAN += loss_GAN_i

            # Feature Matching loss
            loss_FM_i = 0
            for j in range(len(D_fake) - 1):
                loss_FM_i += F.mean_absolute_error(D_fake[i][j], D_real[i][j])
            loss_FM_i = alpha * (1.0 / self.D_num) * loss_FM_i
            chainer.report({'loss_FM'+str(i): loss_FM_i}, gen)
            loss_FM += loss_FM_i

        chainer.report({'loss_GAN': loss_GAN, 'loss_FM': loss_FM}, gen)
        return loss_GAN + loss_FM

    def loss_dis(self, dis, D_fake, D_real):
        loss = 0
        for i in range(self.D_num):
            b, _, h, w = D_real[i][-1].shape
            loss_fake = F.sum(F.softplus(D_fake[i][-1]))/(b*h*w)
            loss_real = F.sum(F.softplus(-D_real[i][-1]))/(b*h*w)
            loss_i = loss_fake + loss_real
            chainer.report({'loss'+str(i): loss_i}, dis)
            loss += loss_i
        chainer.report({'loss_GAN': loss}, dis)
        return loss


    def update_core(self):
        self._iter += 1

        # batch = [oneHot_label_map, edge_map, real_image, feature_map]
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        label = []
        truth = []
        inst = []
        for i in range(batchsize):
            label.append(batch[i][0].data)
            truth.append(np.asarray(batch[i][1]).astype("f"))
            inst.append(np.asarray(batch[i][2]).astype("f"))
        label = Variable(self.xp.asarray(label))
        truth = Variable(self.xp.asarray(truth))
        inst = Variable(np.asarray(inst))

        inst_mean = self.enc(truth, inst)
        gen_input = F.concat((label, inst_mean), axis=1)
        output = self.gen(gen_input)

        fake_input = F.concat((output, label), axis=1)
        real_input = F.concat((truth, label), axis=1)
        D_fake = self.dis(fake_input)
        D_real = self.dis(real_input)

        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen_optimizer.update(self.loss_gen, self.gen, output, D_fake, D_real)
        output.unchain_backward()
        dis_optimizer.update(self.loss_dis, self.dis, D_fake, D_real)


class GlobalUpdater3(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen= kwargs.pop('models')
        self._iter = 0
        self.D_num = 3
        self.xp = self.gen.xp
        self.size = kwargs.pop('size')
        super(GlobalUpdater3, self).__init__(*args, **kwargs)


    def loss_gen(self, gen, output, truth, alpha=10):
        loss_L1 = F.mean_absolute_error(output, truth)
        chainer.report({'loss_L1': loss_L1}, gen)
        return loss_L1

    def update_core(self):
        self._iter += 1

        # batch = [oneHot_label_map, edge_map, real_image, feature_map]
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        label = []
        truth = []
        for i in range(batchsize):
            label.append(batch[i][0].data)
            truth.append(np.asarray(batch[i][1]).astype("f"))
        label = Variable(self.xp.asarray(label))
        truth = Variable(self.xp.asarray(truth))

        output = self.gen(label)

        gen_optimizer = self.get_optimizer('gen')
        gen_optimizer.update(self.loss_gen, self.gen, output, truth)

