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
        self.momentum = params["momentum"]
        self.weights = []
        self.delta = None
        x = self.num_inputs
        for i in range(self.num_hidden + 1):
            if (i == self.num_hidden):
                y = self.num_outputs
            else:
                y = self.hidden_param[i]
            temp = (np.random.rand(x, y) * 4) - 2.0
            self.weights.append(temp)
            x = y

    def sigmoid(self, inp): 
        return 1.0/(1.0 + np.exp(-inp))

    def forprop(self, inp):
        impulse = []
        newinp = np.append(inp, 1.0)
        impulse.append(newinp)
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
            if self.delta != None:
                delta_arr[i] = (delta_arr[i] + 
                        self.momentum * self.delta[i])
            self.weights[i] = self.weights[i] - delta_arr[i]
        self.delta = delta_arr

    def train(self, obs, rating):
        for i in range(len(obs)):
            impulse = self.forprop(obs[i])
            self.backprop(impulse, rating[i])

    def test(self, obs, rating):
        error = 0.0
        for i in range(len(obs)):
            result = self.forprop(obs[i])[self.num_hidden + 1]
            #error += np.inner((result - rating[i]), (result - rating[i]))
            binSize = 1./(self.num_outputs)
            currentValue = binSize/2.
            bins = [0]*self.num_outputs
            for i in range(0, self.num_outputs):
                bins[i] = currentValue
                currentValue = currentValue + binSize
            bins = np.array(bins)
            number = np.average(bins, weights=result) * 10.0
            desired = rating[i]#(scipy.stats.norm.fit(rating[i])[0]) * 10.0
            # print (number)
            # print (desired)
            error += (desired - number)*(desired - number)
        return error/(float(len(obs)))

    def user(self, obs):
        result = self.forprop(obs)[self.num_hidden + 1]
        binSize = 1./(self.num_outputs)
        currentValue = binSize/2.
        bins = [0]*self.num_outputs
        for i in range(0, self.num_outputs):
            bins[i] = currentValue
            currentValue = currentValue + binSize
        bins = np.array(bins)
        number = np.average(bins, weights=result) * 10.0
        return number

