import unittest
import random
import numpy as np

import copy

from jsonParser import Parser
from gaussian import Gaussian
import numpy as np
from nn import NeuralNet

class TestNeuralNet(unittest.TestCase):

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
        self.train[i]['gross'] = self.gauss.gaussian(self.train[i]['gross'], 0.1, 5)
        self.train[i]['budget'] = self.gauss.gaussian(self.train[i]['budget'], 0.1, 5)
        self.train[i]['num_voted_users'] = self.gauss.gaussian(self.train[i]['num_voted_users'], 0.1, 5)
        self.train[i]['num_facebook_like'] = self.gauss.gaussian(self.train[i]['num_facebook_like'], 0.1, 5)
        self.train[i]['director_facebook_likes'] = self.gauss.gaussian(self.train[i]['director_facebook_likes'], 0.1, 5)
        self.train[i]['imdb_score'] = self.gauss.gaussian(self.train[i]['imdb_score'], 0.1, 20)

        # Create actual input numpy matrix
        self.nptrainInp[i] = np.array(self.train[i]['gross'] + self.train[i]['budget'] + self.train[i]['num_voted_users'] + self.train[i]['num_facebook_like'] 
            + self.train[i]['director_facebook_likes'])
        self.nptrainOut[i] = np.array(self.train[i]['imdb_score'])

    for j in range(len(self.crossVal)):
        self.crossVal[j]['gross'] = self.gauss.gaussian(self.crossVal[j]['gross'], 0.1, 5)
        self.crossVal[j]['budget'] = self.gauss.gaussian(self.crossVal[j]['budget'], 0.1, 5)
        self.crossVal[j]['num_voted_users'] = self.gauss.gaussian(self.crossVal[j]['num_voted_users'], 0.1, 5)
        self.crossVal[j]['num_facebook_like'] = self.gauss.gaussian(self.crossVal[j]['num_facebook_like'], 0.1, 5)
        self.crossVal[j]['director_facebook_likes'] = self.gauss.gaussian(self.crossVal[j]['director_facebook_likes'], 0.1, 5)

        self.npcrossInp[j] = np.array(self.crossVal[j]['gross'] + self.crossVal[j]['budget'] + self.crossVal[j]['num_voted_users'] + self.crossVal[j]['num_facebook_like'] 
            + self.crossVal[j]['director_facebook_likes'])
        self.npcrossOut[j] = np.array(self.train[j]['imdb_score'])

    for k in range(len(self.test)):
        self.test[k]['gross'] = self.gauss.gaussian(self.test[k]['gross'], 0.1, 5)
        self.test[k]['budget'] = self.gauss.gaussian(self.test[k]['budget'], 0.1, 5)
        self.test[k]['num_voted_users'] = self.gauss.gaussian(self.test[k]['num_voted_users'], 0.1, 5)
        self.test[k]['num_facebook_like'] = self.gauss.gaussian(self.test[k]['num_facebook_like'], 0.1, 5)
        self.test[k]['director_facebook_likes'] = self.gauss.gaussian(self.test[k]['director_facebook_likes'], 0.1, 5)

        self.nptestInp[k] = np.array(self.test[k]['gross'] + self.test[k]['budget'] + self.test[k]['num_voted_users'] + self.test[k]['num_facebook_like'] 
            + self.test[k]['director_facebook_likes'])
        self.nptestOut[k] = np.array(self.test[k]['imdb_score'])

    self.nptrainInp = np.matrix(self.nptrainInp)
    self.nptrainOut = np.matrix(self.nptrainOut)
    self.npcrossInp = np.matrix(self.npcrossInp)
    self.npcrossOut = np.matrix(self.npcrossOut)
    self.nptestInp = np.matrix(self.nptestInp)
    self.nptestOut = np.matrix(self.nptestOut)

  def test_weight_shapes(self):    
    structure = {'num_inputs': 25, 'num_outputs': 20, 'num_hidden': 1, 
            'learning_rate': 0.8, 'hidden_param':[30]}
    candidate = NeuralNet(structure)
    candidate.train(self.nptrainInp, self.nptrainOut)
    temp = candidate.test(self.npcrossInp, self.npcrossOut)
    print temp
  #   cand_weights = candidate.get_weights()

  #   self.assertEqual(cand_weights[0].shape, (3, 5))
  #   self.assertEqual(cand_weights[1].shape, (5, 1))

  # def test_forward_propagate(self):
  #   learning_rate = 0.8
  #   structure = {'num_inputs': 2, 'num_outputs': 1, 'num_hidden': 1}
  #   candidate = NeuralNet(structure, learning_rate)

  #   x = np.array([1, 0])
  #   cand_out = candidate.forward_propagate(x)
  #   # print(cand_out)

  #   # You can do the math to see what the output should be
  #   # and assert it here.

  #   self.assertEqual(cand_out[0], 0.50056529149822393)

  # def test_backward_propagate(self):
  #   learning_rate = 0.8
  #   structure = {'num_inputs': 2, 'num_outputs': 1, 'num_hidden': 1}
  #   candidate = NeuralNet(structure, learning_rate)

  #   cand_weights = candidate.get_weights()

  #   X = np.array([np.array([1, 0])])
  #   Y = np.array([np.array([0])])
  #   candidate.train(X, Y)

  #   cand_weights = candidate.get_weights()

  #   # You can do the math to see what the new weights should be
  #   # and assert them here.
  #   # print(cand_weights[0])

  #   self.assertAlmostEqual(cand_weights[0][0][0], 1.2894012)
  #   self.assertAlmostEqual(cand_weights[0][1][0], 0.02151894)
  #   self.assertAlmostEqual(cand_weights[0][2][0], 1.29479619)
  #   self.assertAlmostEqual(cand_weights[1][0][0], -3.85282176)

  # def test_xor(self):
  #   learning_rate = 0.2
  #   structure = {'num_inputs': 2, 'num_hidden': 2, 'num_outputs': 1}
  #   candidate = NeuralNet(structure, learning_rate)

  #   labeled_data = [
  #       (np.array([0,0]), np.array([0])),
  #       (np.array([0,1]), np.array([1])),
  #       (np.array([1,0]), np.array([1])),
  #       (np.array([1,1]), np.array([0]))
  #   ]

  #   iterations = 15000

  #   trainX, trainY = zip(*labeled_data)
  #   trainX = np.array(trainX)
  #   trainY = np.array(trainY)

  #   candidate.train(trainX, trainY, iterations)

  #   cand_error = candidate.test(trainX, trainY)
  #   print ("XOR Error: ", cand_error)

  # def test_extra(self):
  #   learning_rate = 0.2
  #   structure = {'num_inputs': 3, 'num_hidden': 6, 'num_outputs': 4}
  #   candidate = NeuralNet(structure, learning_rate)

  #   labeled_data = [
  #       (np.array([0,0,0]), np.array([0,0,0,0])),
  #       (np.array([0,0,1]), np.array([0,0,0,1])),
  #       (np.array([0,1,0]), np.array([1,0,1,0])),
  #       (np.array([0,1,1]), np.array([0,0,1,1])),
  #       (np.array([1,0,0]), np.array([1,1,0,0])),
  #       (np.array([1,0,1]), np.array([1,1,0,1])),
  #       (np.array([1,1,0]), np.array([1,0,0,1])),
  #       (np.array([1,1,1]), np.array([1,1,1,1])),
  #   ]

  #   iterations = 15000

  #   trainX, trainY = zip(*labeled_data)
  #   trainX = np.array(trainX)
  #   trainY = np.array(trainY)

  #   candidate.train(trainX, trainY, iterations)

  #   cand_error = candidate.test(trainX, trainY)
  #   print ("Extra Error: ", cand_error)

if __name__ == '__main__':
  testSuite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralNet)
  testRunner = unittest.TextTestRunner(descriptions=True, verbosity=2)

  testResult = testRunner.run(testSuite)
