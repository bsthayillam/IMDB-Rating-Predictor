import unittest
import random
import numpy as np

import copy

from jsonParser import Parser
from gaussian import Gaussian
import numpy as np
from nn import NeuralNet

class TestNeuralNet():

  def setUp(self):
    self.p = Parser()
    self.gauss = Gaussian()
    (self.train, self.crossVal, self.test) = self.p.parse('imdb_output.json')
    self.nptrainInp = [0] * len(self.train)
    self.nptrainOut = [0] * len(self.train)
    self.npcrossInp = [0] * len(self.crossVal)
    self.npcrossOut = [0] * len(self.crossVal)
    self.nptestInp = [0] * len(self.test)
    self.nptestOut = [0] * len(self.test)
  #   random.seed(0)
  #   np.random.seed(0)

  def test_gaussian(self):
    for i in range(len(self.train)):
        self.train[i]['gross'] = self.gauss.gaussian(self.train[i]['gross'], 0.1, 20)
        self.train[i]['budget'] = self.gauss.gaussian(self.train[i]['budget'], 0.1, 20)
        self.train[i]['num_voted_users'] = self.gauss.gaussian(self.train[i]['num_voted_users'], 0.1, 20)
        self.train[i]['num_facebook_like'] = self.gauss.gaussian(self.train[i]['num_facebook_like'], 0.1, 20)
        self.train[i]['director_facebook_likes'] = self.gauss.gaussian(self.train[i]['director_facebook_likes'], 0.1, 20)
        self.train[i]['imdb_score'] = self.gauss.gaussian(self.train[i]['imdb_score'], 0.1, 20)

        # Create actual input numpy matrix
        self.nptrainInp[i] = np.array(self.train[i]['gross'] + self.train[i]['budget'] + self.train[i]['num_voted_users'] + self.train[i]['num_facebook_like'] 
            + self.train[i]['director_facebook_likes'])
        self.nptrainOut[i] = np.array(self.train[i]['imdb_score'])

    for j in range(len(self.crossVal)):
        self.crossVal[j]['gross'] = self.gauss.gaussian(self.crossVal[j]['gross'], 0.1, 20)
        self.crossVal[j]['budget'] = self.gauss.gaussian(self.crossVal[j]['budget'], 0.1, 20)
        self.crossVal[j]['num_voted_users'] = self.gauss.gaussian(self.crossVal[j]['num_voted_users'], 0.1, 20)
        self.crossVal[j]['num_facebook_like'] = self.gauss.gaussian(self.crossVal[j]['num_facebook_like'], 0.1, 20)
        self.crossVal[j]['director_facebook_likes'] = self.gauss.gaussian(self.crossVal[j]['director_facebook_likes'], 0.1, 20)

        self.npcrossInp[j] = np.array(self.crossVal[j]['gross'] + self.crossVal[j]['budget'] + self.crossVal[j]['num_voted_users'] + self.crossVal[j]['num_facebook_like'] 
            + self.crossVal[j]['director_facebook_likes'])
        self.npcrossOut[j] = self.crossVal[j]['imdb_score']

    for k in range(len(self.test)):
        self.test[k]['gross'] = self.gauss.gaussian(self.test[k]['gross'], 0.1, 20)
        self.test[k]['budget'] = self.gauss.gaussian(self.test[k]['budget'], 0.1, 20)
        self.test[k]['num_voted_users'] = self.gauss.gaussian(self.test[k]['num_voted_users'], 0.1, 20)
        self.test[k]['num_facebook_like'] = self.gauss.gaussian(self.test[k]['num_facebook_like'], 0.1, 20)
        self.test[k]['director_facebook_likes'] = self.gauss.gaussian(self.test[k]['director_facebook_likes'], 0.1, 20)

        self.nptestInp[k] = np.array(self.test[k]['gross'] + self.test[k]['budget'] + self.test[k]['num_voted_users'] + self.test[k]['num_facebook_like'] 
            + self.test[k]['director_facebook_likes'])
        self.nptestOut[k] = self.test[k]['imdb_score']

    self.nptrainInp = np.array(self.nptrainInp)
    self.nptrainOut = np.array(self.nptrainOut)
    self.npcrossInp = np.array(self.npcrossInp)
    self.npcrossOut = np.array(self.npcrossOut)
    self.nptestInp = np.array(self.nptestInp)
    self.nptestOut = np.array(self.nptestOut)

  def test_weight_shapes(self):    
    structure = {'num_inputs': 100, 'num_outputs': 20, 'num_hidden': 3, 
            'learning_rate': 0.2, 'hidden_param':[10, 10, 10]}
    candidate = NeuralNet(structure)
    
    candidate.train(self.nptrainInp, self.nptrainOut)
    temp = candidate.test(self.npcrossInp, self.npcrossOut)
    current = temp
    print("Error for Cross-Validation Data")
    count = 0
    while(temp <= current or count < 5):
        current = temp
        candidate.train(self.nptrainInp, self.nptrainOut)
        temp = candidate.test(self.npcrossInp, self.npcrossOut)
        print(temp)
        count+=1

    testError = candidate.test(self.nptestInp, self.nptestOut)
    print("Error for Testing Data")
    print(testError)

if __name__ == '__main__':
  test = TestNeuralNet()
  test.setUp()
  print("Done setup!")
  test.test_gaussian()
  print("Done converting to gaussian!")
  test.test_weight_shapes()
  print("Finished testing!")
  prompt = input("Want to Test on a movie? (Y/N)")
  while (prompt != "N"):
    gross = input("Input a gross amount: ")
    budget = input("Input a budget amount: ")
    num_voted_users = input("Number of voted users: ")
    num_facebook_like = input("Number of Facebook likes for the movie: ")
    director_facebook_likes = input("Number of Facebook likes for the director: ")
    print("IMDb Rating: " + str(test.user_input(gross, budget, num_voted_users, num_facebook_like, director_facebook_likes)))
    prompt = input("Want to Test on another movie? (Y/N)")
