#python3 -m bonch.cifar.nnet
#This is g**gle collab compataable notebook code


#!pip install git+https://github.com/tinygrad/tinygrad.git
from tinygrad import Device
print(Device.DEFAULT)

import numpy as np

from tinygrad import Device, TinyJit

from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn import optim, Conv2d, Linear
from tinygrad.nn.optim import SGD, Adam
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, get_parameters


import os, gzip, tarfile, pickle
from tinygrad.helpers import fetch

def fetch_cifar():
  X_train = Tensor.empty(50000, 3*32*32, device=f'disk:/tmp/cifar_train_x', dtype=dtypes.uint8)
  Y_train = Tensor.empty(50000, device=f'disk:/tmp/cifar_train_y', dtype=dtypes.int64)
  X_test = Tensor.empty(10000, 3*32*32, device=f'disk:/tmp/cifar_test_x', dtype=dtypes.uint8)
  Y_test = Tensor.empty(10000, device=f'disk:/tmp/cifar_test_y', dtype=dtypes.int64)

  if not os.path.isfile("/tmp/cifar_extracted"):
    def _load_disk_tensor(X, Y, db_list):
      idx = 0
      for db in db_list:
        x, y = db[b'data'], np.array(db[b'labels'])
        assert x.shape[0] == y.shape[0]
        X[idx:idx+x.shape[0]].assign(x)
        Y[idx:idx+x.shape[0]].assign(y)
        idx += x.shape[0]
      assert idx == X.shape[0] and X.shape[0] == Y.shape[0]

    print("downloading and extracting CIFAR...")
    fn = fetch('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    tt = tarfile.open(fn, mode='r:gz')
    _load_disk_tensor(X_train, Y_train, [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)])
    _load_disk_tensor(X_test , Y_test,  [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")])
    open("/tmp/cifar_extracted", "wb").close()

  return X_train, Y_train, X_test, Y_test


# Net
Tensor.manual_seed(1337)
class LalaNet:
  np.random.seed(1337)
  def __init__(self):
    self.l1 = Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = Conv2d(64, 128, kernel_size=(3,3))
    self.l4 = Linear(512, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = x.max_pool2d((3,1,1))
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    x = self.l3(x).relu().max_pool2d((2,2))
    return self.l4(x.flatten(1).dropout(0.5))

# Data
X_train, Y_train, X_test, Y_test = fetch_cifar()

X_train, X_test = X_train.to(device=Device.DEFAULT).float(), X_test.to(device=Device.DEFAULT).float()
Y_train, Y_test = Y_train.to(device=Device.DEFAULT), Y_test.to(device=Device.DEFAULT)
X_train = X_train.reshape((-1, 3, 32, 32))
X_test = X_test.reshape((-1, 3, 32, 32))

print(X_train.shape,"\n",X_test.shape,"\n",Y_train.shape,"\n",X_train.shape[0])

batch_size = 128

model = LalaNet()

# acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# print(acc.item())  # ~10% accuracy, as expected from a random model

parameters = get_parameters(model)
print("parameter count", len(parameters))
optimizer = optim.Adam(parameters)

def step():
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optimizer.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optimizer.step()
  return loss

jit_step = TinyJit(step)

for step in range(7000):
  loss = jit_step()
  if step%100 == 0:
    Tensor.training = False
    # print(model(X_test))
    acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
