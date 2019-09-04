from functions import relU
import random


def dot_product(v, m):
	return [sum(w)*sum(v) for w in m]


class Layer:

	def __init__(self, inputs, outputs, activation_function, bias):
		assert type(outputs) == int
		assert outputs > 0
		assert type(inputs) == int
		assert inputs > 0
		assert type(bias) == float
		self.activation_function = activation_function
		self.bias = bias
		self.weights = []
		for _ in range(outputs):
			self.weights.append([random.triangular(-1, 1) for _ in range(inputs)])

	def predict(self, features):
		assert type(features) == list
		return dot_product(features, self.weights)


class InputLayer(Layer):

	def __init__(self, outputs):
		super().__init__(1, outputs, None, 0.)

	def predict(self, features):
		assert type(features) == list
		assert len(features) == len(self.weights)
		return [features[k]*self.weights[k][0] for k in range(len(self.weights))]


class HiddenLayer(Layer):

	def __init__(self, inputs, outputs):
		super().__init__(inputs, outputs, relU, 1.)

	def predict(self, features):
		return [self.activation_function(feature)+self.bias for feature in super().predict(features)]


class OutputLayer(Layer):

	def __init__(self, inputs, outputs):
		super().__init__(inputs, outputs, None, 0.)


g = 10
p0 = [8, 4]

l0 = InputLayer(outputs=2)
#l0.weights = [[5], [6]]
l1 = HiddenLayer(inputs=2, outputs=3)
#l1.weights = [[0.5, -0.2], [0.4, -0.3], [0.8, -1.2]]
l2 = OutputLayer(inputs=3, outputs=1)
#l2.weights = [[-4.5, 1.3, -0.3]]

p1 = l0.predict(p0)
print(p1)
p2 = l1.predict(p1)
print(p2)
p3 = l2.predict(p2)
print(p3)