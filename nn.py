import numpy as np
import random
import scipy

class NeuralNet:
    def __init__(self, params):
        self.lr = params["learning_rate"]
        self.num_inputs = params["num_inputs"] + 1
        self.num_outputs = params["num_outputs"]
        self.num_hidden = params["num_hidden"]
        self.hidden_param = params["hidden_param"]
        self.weights = []
        x = self.num_inputs
        for i in range(self.num_hidden + 1):
            if (i == self.num_hidden):
                y = self.num_outputs
            else:
                y = self.hidden_param[i]
            temp = (np.random.rand(x, y) * 0.5) - 0.25
            self.weights.append(temp)
            x = y

    def sigmoid(self, inp): 
        return 1.0/(1.0 + np.exp(-inp))

    def forprop(self, inp):
        impulse = []
        impulse.append(np.append(x, 1.0))
        for i in range(self.num_hidden + 1):
            temp = self.sigmoid(np.dot(impulse[i], self.weights[i]))
            impulse.append(temp)
        return impulse

    def backprop(self, impulse, desired):
        out = impulse[self.num_hidden + 1]
        delta_arr = []
        diff = out - desired
        for i in range(self.num_hidden, -1, -1):
            deriv = diff*impulse[i + 1]*(1 - impulse[i + 1])
            delta = np.outer(impulse[i], deriv)*self.lr
            delta_arr.append(delta)
            diff = np.dot(self.weights[i], deriv)
        delta_arr = delta_arr[::-1]
        for i in range(self.num_hidden + 1):
            self.weights[i] = self.weights[i] - delta_arr[i]

    def train(self, obs, rating):
        for i in range(len(obs)):
            impulse = self.forprop(obs[i])
            self.backprop(impulse, rating[i])

    def test(self, obs, rating):
        error = 0.0
        for i in range(len(obs)):
            result = self.forprop(obs[i])[self.num_hidden + 2]
            number = (scipy.stats.norm.fit(result)[0]) * 10.0
            error += (rating[i] - number)*(rating[i] - number)
        return error/(float(len(obs)))

