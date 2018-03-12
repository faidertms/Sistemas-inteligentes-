
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from perceptron import Perceptron
from adaline import Adaline


dataset = pd.read_csv("sonar.csv",header=None)
# cria um dataset , gerando um header
X = dataset.iloc[:, 0:60].values
d = dataset.iloc[:, 60].values
# separa X os atributos e d os tipo

#c1 = dataset[(dataset[4] == 'Iris-setosa')].iloc[:, 0:4].values
#c2 = dataset[(dataset[4] == 'Iris-versicolor')].iloc[:, 0:4].values
#c3 = dataset[(dataset[4] == 'Iris-virginica')].iloc[:, 0:4].values

#c1 = dataset[(dataset[2] == 1)].iloc[:, 0:2].values
#c2 = dataset[(dataset[2] == 0)].iloc[:, 0:2].values
#plt.subplot(311)
#plt.plot(c1[:, 0], c1[:, 1], 'ro', c2[:, 0], c2[:, 1], 'bo')
# aqui é os tipos de rocha c1 = r c2 = m
perceptron = Perceptron(X,d,0.5)
teste = perceptron.W
teste2 = perceptron.X
perceptron.trainning()

# wz=perceptron.execute(X)
aq=perceptron.Train

# adaline = Adaline(X,d,0.5)
# teste = adaline.W
# teste2 = adaline.X
# adaline.trainning()

# wz=adaline.execute(X)
# aq=adaline.Train

plt.plot(aq,'go')
plt.plot(aq,'k:', color='orange')
#plt.ylim([0,100])
plt.grid(True)
plt.xlabel("Epocas")
plt.ylabel("Porcentagem de acerto")
plt.show()