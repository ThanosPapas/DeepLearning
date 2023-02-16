import numpy as np
import random
import pandas as pd

class Perceptron:

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.weights = np.array([])
        self.pt = 0.1


    def _train(self):
        for m in range(500):
            if m > 350 and m < 450:
                self.pt = 0.01
            elif m >= 450:
                self.pt = 0.001
            self.correctly_in_first = []
            self.correctly_in_second = []
            self.misclassified_in_first = []
            self.misclassified_in_second = []
            threshold = 1.7

            for i,j in zip(self.x,self.y):
                # first class: above 0 | second class: below 0. Classify to first class if the product is > 0, and to the second if it isnt
                if self.weights@i > 0 and j < threshold:
                    self.misclassified_in_first.append(i)
                elif self.weights@i < 0 and j > threshold:
                    self.misclassified_in_second.append(-i)
                elif self.weights@i > 0 and j > threshold:
                    self.correctly_in_first.append(i)
                elif self.weights@i < 0 and j < threshold:
                    self.correctly_in_second.append(i)
            if self.misclassified_in_first and self.misclassified_in_second:
                self.weights = self.weights - self.pt * (sum(self.misclassified_in_second) + sum(self.misclassified_in_first))
            else:
                 return 100 , "All elements were classified correctly"

        conf_matrix = [[len(self.correctly_in_first), len(self.misclassified_in_first)], [len(self.misclassified_in_second), len(self.correctly_in_second)]]
        df = pd.DataFrame(conf_matrix, index=['Predicted Expensive', 'Predicted Inexpensive'], columns=['Is Expensive', 'Is Inexpensive'])
        print(self.weights.shape)
        return (1 - ((len(self.misclassified_in_first) + len(self.misclassified_in_second)) / self.x.shape[0])) * 100, df




    def fit(self, x, y):
        self.x = x
        self.y =y
        self.weights = np.array([random.uniform(0,1) for i in range(x.shape[1])])
        self.pt = 0.1
        return self._train()

    def test(self, x, y):
        self.x = x
        self.y = y
        self.correctly_in_first = []
        self.correctly_in_second = []
        self.misclassified_in_first = []
        self.misclassified_in_second = []
        threshold = 1.7

        for i, j in zip(self.x, self.y):
            if self.weights @ i > 0 and j < threshold:
                self.misclassified_in_first.append(i)
            elif self.weights @ i < 0 and j > threshold:
                self.misclassified_in_second.append(-i)
            elif self.weights @ i > 0 and j > threshold:
                self.correctly_in_first.append(i)
            elif self.weights @ i < 0 and j < threshold:
                self.correctly_in_second.append(i)

        conf_matrix = [[len(self.correctly_in_first), len(self.misclassified_in_first)], [len(self.misclassified_in_second), len(self.correctly_in_second)]]
        df = pd.DataFrame(conf_matrix, index=['Predicted Expensive', 'Predicted Inexpensive'], columns=['Is Expensive', 'Is Inexpensive'])
        return (1 - ((len(self.misclassified_in_first) + len(self.misclassified_in_second)) / self.x.shape[0])) * 100, df

