import pandas as pd
import torch
import torch.nn as nn
import random






class Population:
   @staticmethod
   def changeLayer(alpha, layer):
      for i in range(layer.weight.data.shape[0]):
         for j in range(layer.weight.data.shape[1]):
            layer.weight.data[i, j] += random.uniform(-0.8, 0.8)*alpha
      return layer

   @staticmethod
   def changeModel(model, alpha):
      layers = [module for module in model.modules() if type(module) == nn.Linear]

      for i in range(len(layers)):
         layers[i] = Population.changeLayer(alpha, layers[i])

      model = nn.Sequential(layers[0],
                            layers[1],
                            nn.Sigmoid())
      return model


   @staticmethod
   def randomizeLayer(layer):
      for i in range(layer.weight.data.shape[0]):
         for j in range(layer.weight.data.shape[1]):
            layer.weight.data[i, j] = random.uniform(-0.7, 0.7)
      return layer



   @staticmethod
   def makeRandomModel():
      # Сколько на вход, размер 1го скрытого слоя, размер 2го скрытого слоя, размер выхода
      n_in, n_h, n_out = 2, 2, 1

      hLayer1 = nn.Linear(n_in, n_h)
      hLayer1 = Population.randomizeLayer(hLayer1)


      outLayer = nn.Linear(n_h, n_out)
      outLayer = Population.randomizeLayer(outLayer)

      # for i in range(n_in):

      # Create a model
      model = nn.Sequential(hLayer1,
                            outLayer,
                            nn.Sigmoid())
      return model
   SIZE = 10
   allPops = []
   results = []

   def save_model(self, ind1, ind2, path, leave = False):
      torch.save(self.allPops[ind1][ind2].state_dict(), 'createdModel.pt')
      print('Saved model')
      if leave:
          exit()

   def __init__(self):
      print('initialized')
      for i in range(self.SIZE, 0, -1):
         self.allPops.append([0] * i)
         self.results.append([0] * i)

      for i in range(len(self.allPops[0])):
         self.allPops[0][i] = Population.makeRandomModel()

   def makeNext(self):
      ind = -1
      for i in range(len(self.allPops)):
         if self.allPops[i][0] == 0:
            ind = i
            break
      if ind == -1:
         torch.save(self.allPops[self.SIZE - 1][0].state_dict(), 'myModel.pt')
         print('Saved good model')
         print('Result: ')
         print(self.results[self.SIZE-1])
         return -1

      minVal = self.results[ind-1][0]
      delInd = 0
      for i in range(len(self.results[ind-1])):
         if self.results[ind-1][i] < minVal:
            minVal = self.results[ind-1][i]
            delInd = i
      already = False
      for i in range(len(self.allPops[ind])):
         if i == delInd:
            already = True

         if already:
            self.allPops[ind][i] = Population.changeModel(self.allPops[ind - 1][i+1], 0.3)
         else:
            self.allPops[ind][i] = Population.changeModel(self.allPops[ind - 1][i], 0.3)









#a = Population()
#torch.save(a.allPops[0][0].state_dict(), 'myModel.pt')
'''layers = [module for module in a.allPops[0][0].modules() if type(module) == nn.Linear]

print(layers[0].weight.data)


a.allPops[0][0] = Population.changeModel(a.allPops[0][0], 0.2)
layers = [module for module in a.allPops[0][0].modules() if type(module) == nn.Linear]

print(layers[0].weight.data)'''
