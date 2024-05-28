#python3 -m bonch.spine.nnet

import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

path = "/home/lalacode/tinygrad/bonch/spine/"

# Net
Tensor.manual_seed(1337)
class TinyNet:
  np.random.seed(1337)
  def __init__(self):
    self.l1 = Linear(6, 25, bias=False)
    self.l2 = Linear(25, 3, bias=False)

  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x)
    return x

net = TinyNet()

opt = Adam([net.l1.weight, net.l2.weight], eps=1e-8)

# Data
inputData = open(path+"data.data", "r").read().splitlines()
#print(inputData[:1])
inputSet = np.empty((0, 6))
labelSet = np.empty((0, 3))
for spine in inputData:
  spine = spine.split(",")
  spine = np.array(spine)
  inputSet = np.append(inputSet, np.array([spine[:6]], dtype=float), axis=0)
  diagnosis = []
  if ((spine[6]) == "DH"):
     diagnosis = [1,0,0]
  elif ((spine[6]) == "SL"):
     diagnosis = [0,1,0]
  else:
     diagnosis = [0,0,1]
  labelSet = np.append(labelSet, np.array([diagnosis], dtype=float), axis=0)
# shuffle dataset so labels would be spreaded equally
randomSamp = np.random.randint(0, inputSet.shape[0], size=inputSet.shape[0])
trainingDataSet, testingDataSet = np.split(inputSet[randomSamp], [250])
trainingLabelSet, testingLabelSet = np.split(labelSet[randomSamp], [250])

# print(inputSet[:2],labelSet[:2])
# print(trainingDataSet[:1],trainingDataSet.shape[1] ,testingDataSet[:1],trainingLabelSet[:1],trainingLabelSet.shape[1],testingLabelSet[:1])

def trainNet(epochs = 200):
    np.random.seed(1337)
    with Tensor.train():
        for step in range(epochs):
            # random sample a batch
            batchSize = 25
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

            if step % 50 == 0:
                print(f"Step {step+1} | Loss: {loss.numpy():.4f} | Accuracy: {acc:.4f} ")

try:
  state_dict = safe_load(path+"trained1.safetensors")
  load_state_dict(net, state_dict)
  print('Loaded weights "'+path+"trained1.safetensors"+'", evaluating...')
except:
  print('could not load weights "'+path+"trained1.safetensors"+'".')
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
    safe_save(state_dict, path+"trained1.safetensors")
else:
   trainNet(100)
   testNet()
   state_dict = get_state_dict(net)
   safe_save(state_dict, path+"trained1.safetensors")