#!/usr/bin/env python

import chainer
import os
import argparse

from chainer import optimizers, serializers, training
from chainer.training import extensions

from networks import GlobalGenerator, MultiscaleDiscriminator, Encoder
from updater import GlobalUpdater
from dataset import Pix2PixHDDataset

#chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser(
        description='chainer pip2pixHD')
    parser.add_argument('--batchsize', '-b', type=int, default=1)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--dataset', '-i', default="/mnt/sakuradata10-striped/gao/cityscapes")
    parser.add_argument('--out', '-o', default='/mnt/sakuradata10-striped/gao/results/pix2pixHD_ins')
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--snapshot_interval', type=int, default=10000)
    parser.add_argument('--display_interval', type=int, default=10)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--no_one_hot', action='store_false')
    parser.add_argument('--ins_norm', action='store_true')
    parser.add_argument('--vis_num', type=int, default=4)
    parser.add_argument('--vis_interval', type=int, default=100)
    parser.add_argument('--model_num', '-n', default='')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    size = [args.size, args.size * 2]

    gen = GlobalGenerator(ins_norm=args.ins_norm, input_size=size)
    dis = MultiscaleDiscriminator()
    if args.model_num:
        chainer.serializers.load_npz(os.path.join(args.out, 'gen_iter_' + args.model_num + '.npz'), gen)
        chainer.serializers.load_npz(os.path.join(args.out, 'gen_dis_iter_' + args.model_num + '.npz'), dis)

    train = Pix2PixHDDataset(root=args.dataset, one_hot=args.no_one_hot, size=size)
    test = Pix2PixHDDataset(root=args.dataset, one_hot=args.no_one_hot, size=size, test=True)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, shuffle=False)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.0002)
    opt.setup(gen)
    opt_d = optimizers.Adam(alpha=0.0002)
    opt_d.setup(dis)

    # Set up a trainer
    updater = GlobalUpdater(
        models=(gen, dis),
        iterator={'main': train_iter, 'test': test_iter},
        optimizer={'gen': opt, 'dis': opt_d},
        device=args.gpu,
        size=size
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    if args.resume:
        chainer.serializers.load_npz(os.path.join(args.out, 'snapshot_iter_'+args.resume+'.npz'), trainer)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    vis_interval = (args.vis_interval, 'iteration')
    trainer.extend(extensions.dump_graph('gen/loss_GAN'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(args.display_interval, 'iteration'), ))
    report = ['epoch', 'iteration', 'gen/loss_GAN', 'gen/loss_FM', 'dis/loss_GAN']
    trainer.extend(extensions.PrintReport(report))
    trainer.extend(extensions.ProgressBar(update_interval=args.display_interval))
    trainer.extend(train.visualizer(n=args.vis_num, one_hot=args.no_one_hot), trigger=vis_interval)

    trainer.run()

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), gen)
    chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), opt)



if __name__ == '__main__':
    main()

