import pandas as pd
import torch
import torch.nn as nn
import random


def randomizeLayer(layer):
    for i in range(layer.weight.data.shape[0]):
        for j in range(layer.weight.data.shape[1]):
            layer.weight.data[i, j] = random.random()
    return layer

# Сколько на вход, размер 1го скрытого слоя, размер 2го скрытого слоя, размер выхода
n_in, n_h1, n_h2,  n_out = 4, 3, 2, 1


hLayer1 =  nn.Linear(n_in, n_h1)
hLayer1 = randomizeLayer(hLayer1)

hLayer2 = nn.Linear(n_h1, n_h2)
hLayer2 = randomizeLayer(hLayer2)

outLayer = nn.Linear(n_h2, n_out)
outLayer = randomizeLayer(outLayer)

#for i in range(n_in):


# Create a model
model = nn.Sequential(hLayer1,
                      nn.ReLU(),
                      hLayer2,
                      nn.ReLU(),
                      outLayer,
                      nn.Hardtanh())
print(model.__str__())


