#!/usr/bin/env python3

import numpy as np
import chainer
from chainer import serializers
from myfcn import MyFcn
# from chainer import cuda, optimizers, Variable
from chainer import optimizers, Variable
import sys
import math
import time
import mini_batch_loader

MODEL_DUMP_PATH            = "loto6_30_{i}.model"
STATE_DUMP_PATH            = "loto6_30_{i}.state"

#_/_/_/ training parameters _/_/_/
LEARNING_RATE    = 0.0001
TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE  = 100
# TRAIN_BATCH_SIZE = 101
# TEST_BATCH_SIZE  = 101
EPOCHS           = 1000
DECAY_FACTOR     = 0.97
# DECAY_FACTOR     = 1
#DECAY_FACTOR_2   = 0.8
SNAPSHOT_EPOCHS  = 10
EPOCH_BORDER     = 15

def test(loader, model):
    sum_accuracy = 0
    sum_loss     = 0
    test_data_size = loader.count_test_paths()
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x, raw_t = loader.load_testing_data(list(range(i, i+TEST_BATCH_SIZE)))
        # x = chainer.Variable(chainer.cuda.to_gpu(raw_x))
        # t = chainer.Variable(chainer.cuda.to_gpu(raw_t))
        x = chainer.Variable(raw_x)
        t = chainer.Variable(raw_t)
        model.train = False
        model(x, t)
        sum_loss     += model.loss.data * TEST_BATCH_SIZE
        sum_accuracy += model.accuracy * TEST_BATCH_SIZE

    print(("test mean loss {a}, accuracy {b}".format(a=sum_loss/test_data_size, b=sum_accuracy/test_data_size)))
    sys.stdout.flush()

def main():
    loader = mini_batch_loader.MiniBatchLoader()
    # load myfcn model
    myfcn = MyFcn()
    #_/_/_/ setup _/_/_/

    # myfcn = myfcn.to_gpu()
    #optimizer = chainer.optimizers.SGD(LEARNING_RATE)
    #optimizer = chainer.optimizers.MomentumSGD(lr=LEARNING_RATE)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(myfcn)

    #_/_/_/ training _/_/_/
    test(loader, myfcn)
    train_data_size = loader.count_train_paths()
    for epoch in range(1, EPOCHS+1):
        print(("epoch %d" % epoch))
        sys.stdout.flush()

        indices = np.random.permutation(train_data_size)

        sum_accuracy = 0
        sum_loss     = 0

        for i in range(0, train_data_size, TRAIN_BATCH_SIZE):
            r = indices[i:i+TRAIN_BATCH_SIZE]
            raw_x, raw_y = loader.load_training_data(r)
            # x = Variable(chainer.cuda.to_gpu(raw_x))
            # y = Variable(chainer.cuda.to_gpu(raw_y))
            x = Variable(raw_x)
            y = Variable(raw_y)
            myfcn.zerograds()
            myfcn.train = True
            loss = myfcn(x, y)
            loss.backward()
            optimizer.update()

            if math.isnan(loss.data):
                raise RuntimeError("ERROR in main: loss.data is nan!")

            sum_loss     += loss.data * TRAIN_BATCH_SIZE
            sum_accuracy += myfcn.accuracy * TRAIN_BATCH_SIZE

        print(("train mean loss {a}, accuracy {b}".format(a=sum_loss/train_data_size, b=sum_accuracy/train_data_size)))
        sys.stdout.flush()
        test(loader, myfcn)

        #optimizer.lr *= DECAY_FACTOR # if EPOCH_BORDER > epoch else DECAY_FACTOR_2
        if epoch % SNAPSHOT_EPOCHS == 0:
            serializers.save_npz(MODEL_DUMP_PATH.format(i=epoch), myfcn)
            serializers.save_npz(STATE_DUMP_PATH.format(i=epoch), optimizer)

if __name__ == '__main__':
    try:
        start = time.time()
        main()
        end = time.time()
        print(("{s}[s]".format(s=end - start)))
        print(("{s}[m]".format(s=(end - start)/60)))
        print(("{s}[h]".format(s=(end - start)/60/60)))
    except Exception as error:
        print((error.message))
