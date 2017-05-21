import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class MyChain(Chain):
  def __init__(self):
    super(MyChain, self).__init__(
      l1 = L.Linear(4,3),
      l2 = L.Linear(3,3),
    )

  def __call__(self, x, y):
    fv = self.fwd(x, y)
    loss = F.mean_squared_error(fv, y)
    return loss

  def fwd(self, x, y):
    return F.sigmoid(self.l1(x))

def main():
  x1 = Variable(np.array([1]).astype(np.float32))
  x2 = Variable(np.array([2]).astype(np.float32))
  x3 = Variable(np.array([3]).astype(np.float32))

  z = (x1 - 2 * x2 - 1)**2 + (x2 * x3 - 1)**2 + 1
  print(z.data)

  z.backward()
  print(x1.grad)
  print(x2.grad)
  print(x3.grad)

  x = Variable(np.array([-1], dtype=np.float32))
  print(F.sin(x).data)
  print(F.sigmoid(x).data)

  x = Variable(np.array([-0.5], dtype=np.float32))
  z = F.cos(x)
  print(z.data)
  z.backward()
  print(x.grad)
  print((-1) * F.sin(x).data)

  x = Variable(np.array([-1, 0, 1], dtype=np.float32))
  z = F.sin(x)
  z.grad = np.ones(3, dtype=np.float32)
  z.backward()
  print(x.grad)

  h = L.Linear(3,4)
  print(h.W.data)
  print(h.b.data)

  x = Variable(np.array(range(6)).astype(np.float32).reshape(2,3))
  print(x.data)
  y = h(x)
  print(y.data)

  w = h.W.data
  x0 = x.data
  print(x0.dot(w.T) + h.b.data)

  model = MyChain()
  optimizer = optimizers.SGD()
  optimizer.setup(model)

  model.zerograds()
  loss = model(x, y)
  loss.backward()
  optimizer.update()

if __name__ == "__main__":
  main()