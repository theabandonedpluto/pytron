from base import train, predict
import random


class Node:

	def __init__(self, entries, activation_function=None, bias=0):
		assert type(entries) == int
		assert entries > 0
		self.goal_pred = 0
		self.weights = [random.triangular(-1, 1)]*entries
		self.inputs = [None]*entries
		self.pred = 0
		self.observers = []
		self.activation_function = activation_function
		self.bias = bias

	def set_weight(self, k, value):
		assert k < len(self.weights)
		self.weights[k] = value

	def check_activation(self):
		for i in self.inputs:
			if i is None:
				break
		else:
			self.activate()

	def transmit(self, k, feature):
		assert k < len(self.weights)
		self.inputs[k] = feature
		self.check_activation()

	def transmit_all(self, features):
		assert type(features) == list
		for k in range(len(features)):
			self.transmit(k, features[k])

	def connect(self, neuron, k):
		assert k < len(self.weights)
		self.observers.append(lambda pred: neuron.transmit(k, pred))

	def connect(self, cb):
		self.observers.append(cb)

	def activate(self):
		pred = predict(self.inputs, self.weights)
		if self.activation_function:
			pred = self.activation_function(pred)+self.bias
		self.inputs = [None]*len(self.inputs)
		for cb in self.observers:
			cb(pred)

	def train(self, goal_pred, inputs, times=500, alpha=0.01):
		self.weights = train(inputs, self.weights, goal_pred, times, alpha)