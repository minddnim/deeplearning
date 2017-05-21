import numpy as np

import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers
import chainer.functions as F
from chainer.functions.loss import sigmoid_cross_entropy
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

LUCKY_NUMBER_CNT = 7
PAST_NUMBER = 10
MAX_NUMBER = 43

class LotoNN(chainer.Chain):
  def __init__(self):
    super(LotoNN, self).__init__(
      conv1_1=L.Convolution2D( 10,  20, 3, stride=1, pad=1),
      conv1_2=L.Convolution2D( 20,  20, 3, stride=1, pad=1),
      conv1_3=L.Convolution2D( 20,  20, 3, stride=1, pad=1),
      conv1_4=L.Convolution2D( 20,   1, 3, stride=1, pad=1))

  def __call__(self, x):
    h1 = F.relu(self.conv1_1(x))
    h2 = F.relu(self.conv1_2(h1))
    h3 = F.relu(self.conv1_3(h2))
    h = h1 + h2 + h3
    y = F.relu(self.conv1_4(h))
    return y

def run(inputData, outputData):
  xArray = np.array(inputData)
  yArray = np.array(outputData)
  xTrain = xArray[[i for i in range(len(xArray)) if i % 4 != 0],:]
  yTrain = yArray[[i for i in range(len(yArray)) if i % 4 != 0],:]
  xTest = xArray[[i for i in range(len(xArray)) if i % 4 == 0],:]
  yTest = yArray[[i for i in range(len(yArray)) if i % 4 == 0],:]
  xTrain, xTest = np.vsplit(xArray, [int(len(xArray) * 3.0 / 4.0)])
  yTrain, yTest = np.vsplit(yArray, [int(len(xArray) * 3.0 / 4.0)])

  # model = L.Classifier(LotoNN(), lossfun=sigmoid_cross_entropy.sigmoid_cross_entropy)

  model = L.Classifier(LotoNN())
  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)

  train = [[x, y] for x, y in zip(xTrain, yTrain)]
  test = [[x, y] for x, y in zip(xTest, yTest)]

  # train = tuple_dataset.TupleDataset(xTrain, yTrain)
  # test = tuple_dataset.TupleDataset(xTest, yTest)
  trainIter = chainer.iterators.SerialIterator(train, 100)
  testIter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

  updater = training.StandardUpdater(trainIter, optimizer, device=-1)
  trainer = training.Trainer(updater, (100, 'epoch'), out="result")
  trainer.extend(extensions.Evaluator(testIter, model, device=-1))
  trainer.extend(extensions.LogReport())
  trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
  trainer.extend(extensions.ProgressBar())

  trainer.run()