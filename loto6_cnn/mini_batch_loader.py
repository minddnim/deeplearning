#!/usr/bin/env python3

import collections
import os
import numpy as np
import pickle
import test_data as td
import glob
import PIL.Image
import skimage.color
from sklearn.cross_validation import train_test_split

PathInfo = collections.namedtuple('PathInfo', ['image_path', 'label_path'])

#IMAGE_DIR="/home/insight/work/sws/"
IMAGE_DIR="/home/mind/work/pythonStudy/deeplearning/loto6_cnn/data_img_10/"
TRAIN_IMAGE_PATH=IMAGE_DIR + "org"
TARGET_IMAGE_PATH=IMAGE_DIR + "correct"

NUMBER = 1124
PAST_NUMBER = 30
IMG_CNT = NUMBER - PAST_NUMBER

class MiniBatchLoader(object):

    def __init__(self):

        # load a mean image
        # self.mean = np.array([103.939, 116.779, 123.68])
        #self.in_size = (240, 320)
        self.in_size = (10, 5)

    #def load_training_data(self, mini_batch_size):
    #    return self.load_data(mini_batch_size)
        self.train_paths = []
        self.test_paths = []
        train_num = int(IMG_CNT * 3.0 / 4.0)
        for idx in range(train_num):
          self.train_paths.append((idx + 1, TARGET_IMAGE_PATH + "/{i}.png".format(i=(idx + 1))))

        for idx in range(train_num, IMG_CNT):
          self.test_paths.append((idx + 1, TARGET_IMAGE_PATH + "/{i}.png".format(i=(idx + 1))))

    def load_training_data(self, indices):
        return self.load_data(self.train_paths, indices)

    def load_testing_data(self, indices):
        return self.load_data(self.test_paths, indices)

    def count_train_paths(self):
        return len(self.train_paths)

    def count_test_paths(self):
        return len(self.test_paths)

    # test ok
    def load_data(self, paths, indices):

        def image_to_labels(image):
            src = np.asarray(image[:,:,:3])
            ary = np.zeros(src.shape[:2], dtype=np.int32)
            for c, l in [([0,   0,   0],   0),
                         ([255,   0,   0], 1),
                         ([0,   0,   255], 2),
                         ([255, 255, 255], -1)]:
                ary[(src == c).all(2)] = l
            return ary

        mini_batch_size = len(indices)
        in_channels = PAST_NUMBER
        xs = np.zeros((mini_batch_size, in_channels,  *self.in_size)).astype(np.float32)
        ys = np.zeros((mini_batch_size, *self.in_size)).astype(np.int32)

        for i, index in enumerate(indices):
            idx = index % len(paths)
            img_start_idx, label = paths[idx]

            imgs = np.zeros((in_channels,  *self.in_size)).astype(np.float32)
            for img_idx in range(img_start_idx, img_start_idx + PAST_NUMBER):
              path = TRAIN_IMAGE_PATH + "/{i}.png".format(i=img_idx)

              img = np.array(PIL.Image.open(path))
              if img is None:
                raise RuntimeError("invalid image: {i}".format(i=path))

              imgs[(img_idx-img_start_idx), :, :] = image_to_labels(img)

            xs[i, :, :, :] = imgs

            img = np.array(PIL.Image.open(label))
            if img is None:
                raise RuntimeError("invalid image: {i}".format(i=label))
            y = image_to_labels(img)
            # ys[i, :, :] = y[0:y.shape[0]:3, 0:y.shape[1]:4]
            ys[i, :, :] = y

        return xs, ys

def main():
    loader =  MiniBatchLoader()

if __name__ == '__main__':
    main()
