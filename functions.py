import math

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def relU(x):
	return max(0, x)

def relU_prime(x):
    return x > 0