#!/bin/bash -e
#python3 -m bonch.abalone.nnet

import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD, Adam
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

# Net
Tensor.manual_seed(1337)
class TinyNet:
  np.random.seed(1337)
  def __init__(self):
    self.l1 = Linear(8, 25, bias=False)
    self.l2 = Linear(25, 3, bias=False)

  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x)
    return x

net = TinyNet()

opt = Adam([net.l1.weight, net.l2.weight], eps=1e-8)

# Data
inputData = open("/home/lalacode/tinygrad/bonch/lab2/abalone.data", "r").read().splitlines()
#print(inputData[:1])
inputSet = np.empty((0, 8))
labelSet = np.empty((0, 3))
for abalone in inputData:
  abalone = abalone.split(",")
  #convert M,F,I to codes
  abalone[0] = ord(abalone[0])
  abalone = np.array(abalone, dtype=float)
  inputSet = np.append(inputSet, np.array([abalone[:8]]), axis=0)
  # Rings		integer			+1.5 gives the age in years
  age = []
  if ((abalone[8:]+1.5) < 8):
     age = [1,0,0]
  elif ((abalone[8:]+1.5) < 11):
     age = [0,1,0]
  else:
     age = [0,0,1]
  labelSet = np.append(labelSet, np.array([age]), axis=0)

trainingDataSet, testingDataSet = np.split(inputSet, [3000])
trainingLabelSet, testingLabelSet = np.split(labelSet, [3000])

# print(inputSet[:2],labelSet[:2])
# print(trainingDataSet[:1],trainingDataSet.shape[1] ,testingDataSet[:1],trainingLabelSet[:1],trainingLabelSet.shape[1],testingLabelSet[:1])

def trainNet(epochs = 2500):
    np.random.seed(1337)
    with Tensor.train():
        for step in range(epochs):
            # random sample a batch
            batchSize = 150
            samp = np.random.randint(0, trainingDataSet.shape[0], size=(batchSize))
            # print(samp, trainingDataSet[samp])
            batch = Tensor(trainingDataSet[samp], requires_grad=False)
            # get the corresponding labels
            labels = Tensor(trainingLabelSet[samp])

            # forward pass
            out = net(batch)

            # print(out[0].numpy(), labels[0].numpy())
            # compute loss
            loss = Tensor.binary_crossentropy_logits(out, labels)

            # zero gradients
            opt.zero_grad()

            # backward pass
            loss.backward()

            # update parameters
            opt.step()

            # calculate accuracy
            pred = out.argmax(axis=-1).numpy()
            acc = (pred == labels.argmax(axis=-1).numpy()).mean()

            if step % 200 == 0:
                print(f"Step {step+1} | Loss: {loss.numpy():.4f} | Accuracy: {acc:.4f} ")


path = "/home/lalacode/tinygrad/bonch/lab2/trained1.safetensors"
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
    testCycles = 100
    for step in range(testCycles):
      # random sample a batch
      samp = np.random.randint(0, testingDataSet.shape[0], size=(150))
      batch = Tensor(testingDataSet[samp], requires_grad=False)
      # get the corresponding labels
      labels = testingLabelSet[samp]

      # forward pass
      out = net(batch)

      # calculate accuracy
      pred = out.argmax(axis=-1).numpy()
      avg_acc += (pred == labels.argmax(axis=-1)).mean()
    print(f"Test Accuracy: {avg_acc / testCycles}")
    return avg_acc / testCycles

accuracy = testNet()

if accuracy > 0.70:
    state_dict = get_state_dict(net)
    safe_save(state_dict, path)
else:
   trainNet(100)
   testNet()
   state_dict = get_state_dict(net)
   safe_save(state_dict, path)

