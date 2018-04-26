import os

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import cv2
import numpy as np
import pickle

from chainer.dataset import dataset_mixin
import chainer.functions as F


class Pix2PixHDDataset(dataset_mixin.DatasetMixin):

    def __init__(self, root, dtype=np.float32, one_hot=True, no_instmap=False, size=512):
        pairs = []

        f = open(os.path.join(root, 'dataset.txt'), 'r')
        data_paths = f.readlines()

        for i in range(len(data_paths)):
            label, gt = data_paths[i][:-1].split(', ')
            pairs.append((label, gt))

        self._pairs = pairs
        self._root = root
        self._dtype = dtype
        self.crop_size = size
        self.one_hot = one_hot
        self.label_nc = 35
        self.no_instmap = no_instmap

    def __len__(self):
        return len(self._pairs)

    def read_image_as_array(self, path, color):
        img = cv2.imread(path, color)
        img = np.asarray(img, dtype=self._dtype)
        return img

    def get_edges(self, t):
        edge = np.zeros(t.shape).astype(np.uint64)
        edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
        edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
        return edge.astype(self._dtype)

    def get_example(self, i):
        label, gt = self._pairs[i]

        inst = label.replace("_color.png", "_instanceTrainIds.pkl")
        if self.one_hot:
            label = label.replace('_color.png', '_labelIds.png')

        full_path_label = os.path.join(self._root, label)
        full_path_gt = os.path.join(self._root, gt)
        full_path_inst = os.path.join(self._root, inst)
        if self.one_hot:
            img_l = self.read_image_as_array(full_path_label, cv2.IMREAD_GRAYSCALE)
        else:
            img_l = self.read_image_as_array(full_path_label, cv2.IMREAD_COLOR)
        img_g = self.read_image_as_array(full_path_gt, cv2.IMREAD_COLOR)
        img_i = pickle.load(open(full_path_inst, mode='rb'))
        img_e = self.get_edges(img_i)
        img_i = img_i.astype(self._dtype)

        # flip
        if np.random.random() > 0.5:
            img_l = cv2.flip(img_l, 1)
            img_g = cv2.flip(img_g, 1)
            img_e = cv2.flip(img_e, 1)
            img_i = cv2.flip(img_i, 1)
        if np.random.random() > 0.9:
            img_l = cv2.flip(img_l, 0)
            img_g = cv2.flip(img_g, 0)
            img_e = cv2.flip(img_e, 0)
            img_i = cv2.flip(img_i, 0)

        # random down sampling
        scale = np.random.choice(range(20, 40)) / 20
        row = int(img_l.shape[0] // scale)
        col = int(img_l.shape[1] // scale)
        if row >= self.crop_size and col >= self.crop_size:
            s0 = row
            s1 = col
        elif row <= col:
            s0 = self.crop_size
            s1 = int(col / row * self.crop_size)
        elif row > col:
            s0 = int(row / col * self.crop_size)
            s1 = self.crop_size
        img_l = cv2.resize(img_l, (s1, s0), interpolation=cv2.INTER_AREA)
        img_g = cv2.resize(img_g, (s1, s0), interpolation=cv2.INTER_AREA)
        img_e = cv2.resize(img_e, (s1, s0), interpolation=cv2.INTER_AREA)
        img_i = cv2.resize(img_i, (s1, s0), interpolation=cv2.INTER_AREA)

        # crop
        x = np.random.randint(0, img_l.shape[1] - self.crop_size + 1)
        y = np.random.randint(0, img_l.shape[0] - self.crop_size + 1)
        img_l = img_l[y:y + self.crop_size, x:x + self.crop_size]
        img_g = img_g[y:y + self.crop_size, x:x + self.crop_size]
        img_e = img_e[y:y + self.crop_size, x:x + self.crop_size]
        img_i = img_i[y:y + self.crop_size, x:x + self.crop_size]

        if self.one_hot:
            input_label = np.zeros((self.label_nc, img_l.shape[0], img_l.shape[1]))
            img_l = img_l.astype('i')
            ids = np.unique(img_l)
            for id in ids:
                input_label[id, :] = img_l == id
            # for id in ids:
            #     input_label[id] = np.where(img_l==id, 1, 0)
            img_l = input_label.astype(self._dtype)
        else:
            img_l = img_l.transpose(2, 0, 1)

        img_e = img_e[np.newaxis, :, :] * 255
        img_i = img_i[np.newaxis, :, :]

        if not self.no_instmap:
            img_l = F.concat((img_l, img_e), axis=0)

        #img_l = img_l.transpose(2, 0, 1)

        img_g = img_g.transpose(2, 0, 1) / 255.0 # * 2 - 1
        img_i = img_i.transpose(2, 0, 1)

        return img_l, img_g, img_i
