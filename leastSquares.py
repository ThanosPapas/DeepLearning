import numpy as np
class LSquares:

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.weights = np.array([])

    def fit(self, x, y):
        '''Optimize the weights and then return the MSE and MAE'''

        self.x=x
        self.y=y
        self.weights = np.linalg.inv(self.x.T@self.x)@self.x.T@self.y

        predicted_values = [self.weights.T@i for i in self.x]
        s = 0
        a = 0
        for i,j in zip(predicted_values, self.y):
              s+=pow(i - j, 2)
              a+=abs(i - j)
        mse = s / self.x.shape[0]
        mae = a / self.x.shape[0]
        return mse,mae