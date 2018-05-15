import os
import random
import cv2
import numpy as np
import chainer
from chainer.dataset import dataset_mixin
from chainercv.transforms import flip

try:
    from cityscapesScripts.cityscapesscripts.helpers.labels import labels, id2label
except ImportError:
    raise ImportError("citysscapescripts not in path. See https://github.com/mcordts/cityscapesScripts")



class Pix2PixHDDataset(dataset_mixin.DatasetMixin):

    def __init__(self, root, dtype=np.float32, one_hot=True, size=(512, 1024), test=False, use_edge=True):

        self.size = size
        self.one_hot = one_hot
        self.input_channels = 35
        self.test = test
        self.use_edge = use_edge
        self.pairs = self.build_list(root)


    def build_list(self, root):
        _root = os.path.join(root, "leftImg8bit", "train" if not self.test else "val")
        _gtroot = os.path.join(root, "gtFine", "train" if not self.test else "val")

        if not isinstance(_root, (list, tuple)):
            _root = [_root]
        if not isinstance(_gtroot, (list, tuple)):
            _gtroot = [_gtroot]

        pairs = []

        for rootpath, gtrootpath in zip(_root, _gtroot):
            for seq in os.listdir(rootpath):
                path = os.path.join(rootpath, seq)
                gtpath = os.path.join(gtrootpath, seq)
                for img in os.listdir(path):
                    fullpath = os.path.join(path, img)
                    truth_file = "gtFine_labelIds" if self.one_hot else "gtFine_color"
                    gtfullpath = os.path.join(gtpath, img.replace("leftImg8bit", truth_file))
                    egfullpath = os.path.join(gtpath, img.replace("leftImg8bit", "gtFine_edge"))
                    pairs.append((fullpath, gtfullpath, egfullpath))

        print("found {} pairs".format(len(pairs)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def get_image(self, imagepath):
        h, w = self.size
        img = cv2.imread(imagepath, cv2.IMREAD_COLOR).astype(np.float32)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img /= 255.0
        img *= 2.0
        img -= 1.0
        return img.transpose(2, 0, 1)

    def get_label(self, imagepath):
        h, w = self.size
        img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED if self.one_hot else cv2.IMREAD_COLOR).astype(np.float32)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.one_hot:
            img = self.onehot_labels(img)
        else:
            img /= 255.0
            img *= 2.0
            img -= 1.0
        return img.transpose(2, 0, 1)

    def onehot_labels(self, lbl):
        _lbl = np.zeros((lbl.shape + (self.input_channels,))).astype(np.uint8)
        for i in range(self.input_channels):
            _lbl[:,:, i] = lbl==i
        return _lbl.astype(np.float32)

    def get_edge(self, imagepath):
        h, w = self.size
        img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        if self.one_hot:
            img = img / 255.0
        else:
            img /= 255.0
            img *= 2.0
            img -= 1.0
        return img[np.newaxis, :, :]

    def get_example(self, i):
        imgpath, lblpath, edgepath = self.pairs[i]

        img = self.get_image(imgpath)
        lbl = self.get_label(lblpath)
        edg = self.get_edge(edgepath)

        if random.random() < 0.5:
            lbl = flip(lbl, x_flip=True)
            img = flip(img, x_flip=True)
            edg = flip(edg, x_flip=True)

        if self.use_edge:
            lbl = np.concatenate([lbl, edg], axis=0)
        return lbl, img

    def transform_image(self, img):
        img = img.transpose(1, 2, 0)
        img += 1
        img /= 2
        img *= 255.0
        return img

    def visualizer(self, output_path="preview", n=1, one_hot=False):
        @chainer.training.make_extension()
        def make_image(trainer):
            updater = trainer.updater
            output = os.path.join(trainer.out, output_path)
            os.makedirs(output, exist_ok=True)

            rows = []
            for i in range(n):
                label, image = updater.converter(updater.get_iterator("test").next(), updater.device)

                # turn off train mode
                with chainer.using_config('train', False):
                    generated = updater.get_optimizer("gen").target(label).data

                # convert to cv2 image
                img = self.transform_image(generated[0])
                label = label[0].transpose(1, 2, 0)
                image = self.transform_image(image[0])

                # return image from device if necessary
                if updater.device >= 0:
                    img = img.get()
                    label = label.get()
                    image = image.get()

                # convert the onehot label to RGB
                if one_hot:
                    _label = np.zeros_like(img).astype(np.float32)
                    for i, lbl in enumerate(labels):
                        if lbl.ignoreInEval is False:
                            mask = label[: ,:, lbl.id] == 1
                            _label[mask] = np.array(lbl.color[::-1])
                else:
                    _label = label
                    _label += 1
                    _label /= 2
                    _label *= 255.0

                rows.append(np.hstack((_label, img, image)).astype(np.uint8))
                cv2.imwrite(os.path.join(output, "iter_{}.png".format(updater.iteration)), np.vstack(rows))

        return make_image
