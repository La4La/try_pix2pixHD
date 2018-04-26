import os
import glob
import numpy as np

import chainer
import chainer.cuda
from chainer import cuda, Variable
import cv2
import pickle
import chainer.functions as F


def test_visualizer(updater, generator, output_path, test_image_path, s_size=512, one_hot=True):
    @chainer.training.make_extension()
    def get_edges(t):
        edge = np.zeros(t.shape).astype(np.uint64)
        edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
        edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
        return edge.astype(np.float32)

    def read_image_as_array(path, color):
        img = cv2.imread(path, color)
        img = np.asarray(img, dtype=np.float32)
        return img
    
    def read_img(path):
        if one_hot:
            inst_path = path.replace("_labelIds.png", "_instanceTrainIds.pkl")
        else:
            inst_path = path.replace("_color.png", "_instanceTrainIds.pkl")
        img_i = pickle.load(open(inst_path, mode='rb'))
        img_e = get_edges(img_i)

        if one_hot:
            img_l = read_image_as_array(path, cv2.IMREAD_GRAYSCALE)
        else:
            img_l = read_image_as_array(path, cv2.IMREAD_COLOR)

        if img_l.shape[0] < img_l.shape[1]:
            s0 = s_size
            s1 = int(img_l.shape[1] * (s_size / img_l.shape[0]))
        else:
            s1 = s_size
            s0 = int(img_l.shape[0] * (s_size / img_l.shape[1]))

        img_l = np.asarray(img_l, np.float32)
        img_l = cv2.resize(img_l, (s1, s0), interpolation=cv2.INTER_AREA)
        img_e = cv2.resize(img_e, (s1, s0), interpolation=cv2.INTER_AREA)

        if one_hot:
            input_label = np.zeros((35, img_l.shape[0], img_l.shape[1]))
            ids = np.unique(img_l).astype('i')
            for id in ids:
                input_label[id] = np.where(img_l==id, 1, 0)
            img_l = input_label.astype(np.float32)
        else:
            img_l = img_l.transpose(2, 0, 1)
        img_e = img_e[np.newaxis, :, :]

        img_l = F.concat((img_l, img_e), axis=0)
        return img_l.data

    def save_as_img(array, name):
        array = array.transpose(1, 2, 0)
        # array = (array + 1) / 2 * 255
        array = array * 255
        array = array.clip(0, 255).astype(np.uint8)
        img = cuda.to_cpu(array)
        cv2.imwrite(name, img)

    def process(file_path, output_path):
        sample = read_img(file_path)
        if one_hot:
            x_in = np.zeros((1, 36, sample.shape[1], sample.shape[2]), dtype='f')
        else:
            x_in = np.zeros((1, 4, sample.shape[1], sample.shape[2]), dtype='f')
        x_in[0, :] = sample
        x_in = cuda.to_gpu(x_in)
        cnn_in = Variable(x_in)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                cnn_out = generator(cnn_in)
        cnn_out = chainer.cuda.to_cpu(cnn_out.data)
        save_as_img(cnn_out[0], output_path)

    def execute(trainer):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if one_hot:
            file_test = glob.glob(test_image_path + '/*_labelIds.png')
        else:
            file_test = glob.glob(test_image_path + '/*_color.png')

        for f in file_test:
            filename = os.path.basename(f)
            filename = os.path.splitext(filename)[0]
            process(f, output_path+"/iter_"+str(trainer.updater.iteration)+"_"+filename+".png")

    return execute