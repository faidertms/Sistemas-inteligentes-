import numpy as np

class Perceptron:
    
    
    def __init__(self, data, d, learning_rate = 0.5):
        self.X = np.ones((len(data),1), dtype=int)
        self.X = np.c_[(self.X,data)]
        self.d = d
        self.learning_rate = learning_rate
        self.epochs = 0
        self.W = np.random.rand(len(self.X[0,:]))
        self.Train = 0
    
    def trainning(self):
        
        error = True
        
        while(error and self.epochs <= 10000):
            
            self.epochs += 1
            error = False
            acerto = 0
            
            for i in range(len(self.X)):
                u = self.X[i,:].dot(self.W)
                y = self.stepFunction(u)
                
                if y != self.d[i]:
                    error = True
                    for j in range(len(self.W)):
                        self.W[j] = self.W[j] + self.learning_rate * (self.d[i] - y) * self.X[i,j]
                else:
                    acerto +=1
                        
            acertoPorct = (acerto/len(self.X))*100
            self.Train = np.append(self.Train, acertoPorct)
            print(self.epochs)
    
    def execute(self, data):
        # cria uma matriz nova para criação
        if np.ndim(data) == 1:
            self.c = np.insert(data,0,1)
            return self.stepFunction(self.c.dot(self.W))
        else:
            self.c = np.ones((len(data),1), dtype=int)
            self.c = np.c_[(self.c,data)]
            return [self.stepFunction(self.W.dot(x)) for x in self.c]
    
    def stepFunction(self, u):
        return 1 if u >= 0 else 0
    