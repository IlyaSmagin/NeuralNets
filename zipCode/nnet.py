#!/bin/bash -e
# CLANG=1 python3 -m bonch.zipCode.nnet
# https://codepen.io/lalacode/full/rNbebeB for visualisation

path = "/home/lalacode/tinygrad/bonch/zipCode/trained1.safetensors"

import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

class TinyNet:
  def __init__(self):
    np.random.seed(1337)
    self.l1 = Linear(9, 10, bias=False)
    self.l2 = Linear(10, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x

net = TinyNet()

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

opt = SGD([net.l1.weight, net.l2.weight], lr=3e-2)

origNumbers = np.array([[1,1,0,1,0,1,0,1,1],[0,0,1,1,0,0,1,0,0],[1,0,0,1,0,0,1,0,1],[1,0,1,0,1,0,1,0,0],[0,1,0,1,1,0,0,1,0],[1,1,0,0,1,0,0,1,1],[0,0,1,0,1,1,0,1,1],[1,0,1,0,0,1,0,0,0],[1,1,0,1,1,1,0,1,1],[1,1,0,1,1,0,1,0,0]])
trainingWithNoise = []
trainingLabels = []
for juk in range(10):
  number = origNumbers[juk]
  trainingWithNoise.append(number)
  trainingLabels.append(juk)
  for bit in range(9):
    noisyNumber = number.copy()
    noisyNumber[bit] = noisyNumber[bit] ^ 1
    trainingWithNoise.append(noisyNumber)
    trainingLabels.append(juk)

trainingWithNoise = np.array(trainingWithNoise)
trainingLabels = np.array(trainingLabels)

#print(trainingWithNoise[96], trainingLabels[0], trainingWithNoise.shape[0], trainingLabels.shape[0])

def trainNet(epochs = 1000):
    with Tensor.train():
        for step in range(epochs):
            # random sample a batch
            batchSize = 20
            samp = np.random.randint(0, trainingWithNoise.shape[0], size=(batchSize))
            # print(samp, trainingWithNoise[samp])
            batch = Tensor(trainingWithNoise[samp], requires_grad=False)
            # get the corresponding labels
            labels = Tensor(trainingLabels[samp])

            # forward pass
            out = net(batch)

            # compute loss
            loss = sparse_categorical_crossentropy(out, labels)

            # zero gradients
            opt.zero_grad()

            # backward pass
            loss.backward()

            # update parameters
            opt.step()

            # calculate accuracy
            pred = out.argmax(axis=-1)
            acc = (pred == labels).mean()

            if step % 10 == 0:
                print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")


try:
  state_dict = safe_load(path)
  load_state_dict(net, state_dict)
  print('Loaded weights "'+path+'", evaluating...')
except:
  print('could not load weights "'+path+'".')
  print("retraining net...")
  trainNet()

def testNet():
  with Timing("Time: "):
    avg_acc = 0
    testCycles = 10
    for step in range(testCycles):
      # random sample a batch
      samp = np.random.randint(0, trainingWithNoise.shape[0], size=(10))
      batch = Tensor(trainingWithNoise[samp], requires_grad=False)
      # get the corresponding labels
      labels = trainingLabels[samp]

      # forward pass
      out = net(batch)

      # calculate accuracy
      pred = out.argmax(axis=-1).numpy()
      avg_acc += (pred == labels).mean()
    print(f"Test Accuracy: {avg_acc / testCycles}")
    return avg_acc / testCycles

accuracy = testNet()

rand = Tensor([1,1,0,1,0,0,0,1,1])
out = net(rand)
pred = out.argmax(axis=-1).numpy()
for number in range(10):
  isMostLikelyNumber = "- result" if number == pred else ""
  print(number, ": ", out.softmax().numpy()[number], isMostLikelyNumber)

if accuracy > 0.9:
    state_dict = get_state_dict(net)
    safe_save(state_dict, path)
else:
   trainNet(100)
   testNet()
   state_dict = get_state_dict(net)
   safe_save(state_dict, path)