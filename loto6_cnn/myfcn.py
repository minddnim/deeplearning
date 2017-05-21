#!/usr/bin/env python3

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from add import add
import numpy as np
import math

class MyFcn(chainer.Chain):
    CLASSES = 3

    def __init__(self):
        super(MyFcn, self).__init__(
            conv1_1=L.Convolution2D( 30,  64, 3, stride=1, pad=1),
            bn1    =F.BatchNormalization(64),
            conv1_2=L.Convolution2D( 64,  64, 3, stride=1, pad=1),
            bn2    =F.BatchNormalization(64),
            conv1_3=L.Convolution2D( 64, 64, 3, stride=1, pad=1),
            bn3    =F.BatchNormalization(64),
            conv1_4=L.Convolution2D( 64, 32, 3, stride=1, pad=1),
            bn4    =F.BatchNormalization(32),
            conv1_5=L.Convolution2D( 32, 32, 3, stride=1, pad=1),
            bn5    =F.BatchNormalization(32),
            conv1_6=L.Convolution2D( 32, 32, 3, stride=1, pad=1),
            bn6    =F.BatchNormalization(32),
            conv1_7=L.Convolution2D( 32, MyFcn.CLASSES, 3, stride=1, pad=1)

            # conv2_1=L.Convolution2D( 64, 128, 3, stride=1, pad=1),
            # conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            # conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            # conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            # conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            # conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            # conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            # conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            # conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            # conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            # conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            # score_pool3=L.Convolution2D(256, MyFcn.CLASSES, 1, stride=1, pad=0),
            # score_pool4=L.Convolution2D(512, MyFcn.CLASSES, 1, stride=1, pad=0),
            # score_pool5=L.Convolution2D(512, MyFcn.CLASSES, 1, stride=1, pad=0),

            # upsample_pool4=L.Deconvolution2D(MyFcn.CLASSES, MyFcn.CLASSES, ksize= 4, stride=2, pad=1),
            # upsample_pool5=L.Deconvolution2D(MyFcn.CLASSES, MyFcn.CLASSES, ksize= 8, stride=4, pad=2),
            # upsample_final=L.Deconvolution2D(MyFcn.CLASSES, MyFcn.CLASSES, ksize=16, stride=8, pad=4),
        )
        self.train = True

    def __call__(self, x, t):

        h1 = F.relu(self.bn1(self.conv1_1(x)))
        h2 = F.relu(self.bn2(self.conv1_2(h1)))
        h3 = F.relu(self.bn3(self.conv1_3(h2)))
        h123 = add(h1, h2, h3)
        h4 = F.relu(self.bn4(self.conv1_4(h123)))
        h5 = F.relu(self.bn5(self.conv1_5(h4)))
        h6 = F.relu(self.bn6(self.conv1_6(h5)))
        h456 = add(h4, h5, h6)
        h = F.relu(self.conv1_7(h456))
        self.final_shape = h.data.shape
        # h = F.max_pooling_2d(h, 2, stride=2)

        # h = F.relu(self.conv2_1(h))
        # h = F.relu(self.conv2_2(h))
        # h = F.max_pooling_2d(h, 2, stride=2)

        # h = F.relu(self.conv3_1(h))
        # h = F.relu(self.conv3_2(h))
        # h = F.relu(self.conv3_3(h))
        # h = F.max_pooling_2d(h, 2, stride=2)
        # pool3 = h

        # h = F.relu(self.conv4_1(h))
        # h = F.relu(self.conv4_2(h))
        # h = F.relu(self.conv4_3(h))
        # h = F.max_pooling_2d(h, 2, stride=2)
        # pool4 = h

        # h = F.relu(self.conv5_1(h))
        # h = F.relu(self.conv5_2(h))
        # h = F.relu(self.conv5_3(h))
        # h = F.max_pooling_2d(h, 2, stride=2)
        # pool5 = h

        # p3 = self.score_pool3(pool3)
        # self.p3_shape = p3.data.shape

        # p4 = self.score_pool4(pool4)
        # self.p4_shape = p4.data.shape

        # p5 = self.score_pool5(pool5)
        # self.p5_shape = p5.data.shape

        # u4 = self.upsample_pool4(p4)
        # self.u4_shape = u4.data.shape

        # u5 = self.upsample_pool5(p5)
        # self.u5_shape = u5.data.shape

        # h = add(p3, u4, u5)

        # h = self.upsample_final(h)
        # self.final_shape = h.data.shape

        self.loss = F.softmax_cross_entropy(h, t)
        if math.isnan(self.loss.data):
            raise RuntimeError("ERROR in MyFcn: loss.data is nan!")
        self.accuracy = self.calculate_accuracy(h, t)
        if self.train:
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

    def calculate_accuracy(self, predictions, truths):
        gpu_predictions = predictions.data
        gpu_truths = truths.data

        cpu_predictions = chainer.cuda.to_cpu(gpu_predictions)
        cpu_truths = chainer.cuda.to_cpu(gpu_truths)

        # we want to exclude labels with -1
        mask = cpu_truths != -1

        # reduce values along classe axis
        reduced_cpu_preditions = np.argmax(cpu_predictions, axis=1)

        # mask
        masked_reduced_cpu_preditions = reduced_cpu_preditions[mask]
        masked_cpu_truths = cpu_truths[mask]

        s = (masked_reduced_cpu_preditions == masked_cpu_truths).mean()
        return s
