#!/usr/bin/env python3

import skimage.io
import numpy as np

import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from myfcn import MyFcn
import PIL.Image

MODEL_DUMP_PATH = "loto6_30_{i}.model"
NUMBER = 1125
PAST_NUMBER = 30
IMAGE_SIZE = (10, 5)
IMAGE_DIR="/home/mind/work/pythonStudy/deeplearning/loto6_cnn/data_img_10/"
TRAIN_IMAGE_PATH=IMAGE_DIR + "org"

def analyze(img):
    myfcn = MyFcn()
    serializers.load_npz(MODEL_DUMP_PATH.format(i="1000"), myfcn)

    (h, w) = IMAGE_SIZE
    myfcn.train = False

    t = np.zeros(IMAGE_SIZE, dtype=np.int32)
    y = myfcn(Variable(np.array([img.astype(np.float32)])),
              Variable(np.asarray([t])))
    [pred] = y.data
    result = np.argmax(pred, axis=0)
    print(result)
    return result

def image_to_labels(image):
    src = np.asarray(image[:,:,:3])
    ary = np.zeros(src.shape[:2], dtype=np.int32)
    for c, l in [([0,   0,   0],   0),
                 ([255,   0,   0], 1),
                 ([0,   0,   255], 2),
                 ([255, 255, 255], -1)]:
        ary[(src == c).all(2)] = l
    return ary

def main():
  import sys

  img_start_idx = NUMBER - PAST_NUMBER
  imgs = np.zeros((PAST_NUMBER, *(10, 5))).astype(np.float32)
  for img_idx in range(img_start_idx, img_start_idx + PAST_NUMBER):
    path = TRAIN_IMAGE_PATH + "/{i}.png".format(i=img_idx)

    img = np.array(PIL.Image.open(path))
    if img is None:
      raise RuntimeError("invalid image: {i}".format(i=path))

    imgs[(img_idx-img_start_idx), :, :] = image_to_labels(img)

  result = analyze(imgs)
  img2 = PIL.Image.fromarray(np.uint8(result))
  img2.save("./{i}.png".format(i=NUMBER))


if __name__ == '__main__':
    main()
