#!/usr/bin/env python

import chainer
import os
import argparse

from chainer import optimizers, serializers, training
from chainer.training import extensions

from networks import GlobalGenerator, MultiscaleDiscriminator, Encoder
from updater import GlobalUpdater, GlobalUpdater2, GlobalUpdater3
from dataset import Pix2PixHDDataset
from training_visualizer import test_visualizer

#chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser(
        description='chainer pip2pixHD')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default="/mnt/sakuradata10-striped/gao/cityscapes",
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='/mnt/sakura201/gao/pix2pixHD/result_IN',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=2,
                        help='Interval of displaying log to console')
    parser.add_argument('--test_visual_interval', type=int, default=100,
                        help='Interval of drawing test images')
    parser.add_argument('--test_out', default='./test_result_IN/',
                        help='DIrectory to output test samples')
    parser.add_argument('--test_image_path', default = '/mnt/sakuradata10-striped/gao/test_samples/test_city',
                        help='Directory of image files for testing')
    parser.add_argument('--size', type=int, default=512,
                        help='Size of the training images')
    parser.add_argument('--one_hot', action='store_true')
    parser.add_argument('--use_encoder', '-enc', action='store_true')
    parser.add_argument('--not_SNGAN', action='store_false')
    parser.add_argument('--ins_norm', action='store_true')
    parser.add_argument('--L1', action='store_true')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    if args.one_hot:
        gen_in_ch_n = 36
        dis_in_ch_n = 39
    else:
        gen_in_ch_n = 4
        dis_in_ch_n = 7
    if args.use_encoder:
        gen_in_ch_n += 3

    gen = GlobalGenerator(in_ch=gen_in_ch_n, ins_norm=args.ins_norm)
    # serializers.load_npz("/mnt/sakura201/gao/pix2pixHD/result4/gen_iter_15000.npz", gen)
    # print('generator loaded')

    if not args.L1:
        dis = MultiscaleDiscriminator(in_ch=dis_in_ch_n, SNGAN=args.not_SNGAN)
        #serializers.load_npz("/mnt/sakura201/gao/lineart/model2/gen_dis_iter_8000", dis)
        #print('discriminator loaded')

    if args.use_encoder:
        enc = Encoder()

    dataset = Pix2PixHDDataset(root=args.dataset, one_hot=args.one_hot, size=args.size)
    iteration = chainer.iterators.SerialIterator(dataset, args.batchsize)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        if not args.L1:
            dis.to_gpu()
        if args.use_encoder:
            enc.to_gpu()

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.0001)
    opt.setup(gen)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_gen')

    if not args.L1:
        opt_d = optimizers.Adam(alpha=0.001)
        opt_d.setup(dis)
        opt_d.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')

    # Set up a trainer
    if args.L1:
        updater = GlobalUpdater3(
            models=(gen),
            iterator={'main': iteration},
            optimizer={'gen': opt},
            device=args.gpu,
            size=args.size
        )
    else:
        updater = GlobalUpdater(
            models=(gen, dis),
            iterator={'main': iteration},
            optimizer={'gen': opt, 'dis': opt_d},
            device=args.gpu,
            size=args.size
        )


    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    # serializers.load_npz('/mnt/sakura201/gao/pix2pixHD/result/snapshot_iter_2000.npz', trainer)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    # trainer.extend(extensions.dump_graph('gen/loss_GAN'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    if not args.L1:
        trainer.extend(extensions.snapshot_object(
            dis, 'gen_dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    if args.use_encoder:
        trainer.extend(extensions.snapshot_object(
            enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(args.display_interval, 'iteration'), ))

    if args.L1:
        report = ['epoch', 'iteration', 'gen/loss_L1']
    else:
        report = ['epoch', 'iteration', 'gen/loss_GAN', 'gen/loss_GAN0', 'gen/loss_GAN1', 'gen/loss_GAN2',
                  'gen/loss_FM', 'dis/loss_GAN']

    trainer.extend(extensions.PrintReport(report))
    trainer.extend(extensions.ProgressBar(update_interval=args.display_interval))
    trainer.extend(test_visualizer(updater, gen, args.test_out, args.test_image_path,
                                   s_size=args.size, one_hot=args.one_hot),
                   trigger=(args.test_visual_interval, 'iteration'))

    trainer.run()

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), gen)
    chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), opt)



if __name__ == '__main__':
    main()

