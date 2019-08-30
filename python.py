def gradient_descent(
	input_value:float,
	weight:float,
	goal_pred:float,
	alpha:float=1.0
)->(float, float): # weight, error
	"""
		Approximate an ideal weight to reach the goal_pred
		according to the given data. Set the right alpha
		value to make better predictions.
	"""
	assert type(input_value) == float
	assert type(weight) == float
	assert type(goal_pred) == float
	assert type(alpha) == float
	prediction = input_value*weight
	pure_error = prediction-goal_pred
	error = pure_error**2
	derivative = pure_error*weight
	return weight-derivative*alpha, error


def repeated_gradient_descent(
	times:int,
	data:float,
	weight:float,
	goal_pred:float,
	alpha:float=1.0
)->(float, float): # weight, error
	"""
		Repeat the gradient descent N times.
	"""
	assert type(times) == int
	assert times > 0
	assert type(data) == float
	assert type(weight) == float
	assert type(goal_pred) == float
	assert type(alpha) == float
	for _ in range(times):
		weight, error = gradient_descent(data, weight, goal_pred, alpha)
	return weight, error


def predict(
	data:list,
	weights:list
)->float: # pred
	"""
		Make a prediction
	"""
	assert len(data) == len(weights)
	assert type(data) == list
	assert type(weights) == list
	return sum([data[k]*weights[k] for k in range(len(data))])


def train(
	data:list,
	weights:list,
	goal_pred:float,
	times:int=20
)->list: # weights
	"""
		Train the neuron to get the best weights for the given data
	"""
	assert type(data) == list
	assert type(weights) == list
	assert len(data) == len(weights)
	assert type(goal_pred) == float
	assert type(times) == int
	output = []
	for k in range(len(data)):
		weight, error = repeated_gradient_descent(times, data[k], weights[k], goal_pred)
		output.append(weight)
	return output

def learn(
	data:list,
	weights:list,
	goal_pred:float,
	times:int=20,
	alpha:float=1.0
)->list: # updated weights
	""" Transform weights to reach the goal prediction """
	assert type(data) == list
	assert type(weights) == list
	assert len(data) == len(weights)
	assert type(goal_pred) == float
	assert type(times) == int
	assert type(alpha) == float
	assert alpha != 0
	for _ in range(times):
		pred = predict(data, weights)
		delta = pred-goal_pred
		weight_deltas = [data[k]*delta for k in range(len(data))]
		weights = [weights[k]-(weight_deltas[k]*alpha) for k in range(len(data))]
	return weights, delta**2

def main():
	data = [1.0, 3.0, 2.0]
	weights = [0.5, 2.0, 0.1]
	goal_pred = -10.0
	alpha = 1e-2
	times = 100
	print(f"g={goal_pred}")
	pred = predict(
		data,
		weights
	)
	print(f"p={pred}")
	print(f"*Training*")
	weights, error = learn(
		data,
		weights,
		goal_pred,
		times,
		alpha
	)
	pred = predict(
		data,
		weights
	)
	print(f"p={pred} | e={error}")
	print(f"confidence={round(100-(abs(pred-goal_pred))*100/(pred+goal_pred), 2)}%")


if __name__ == '__main__':
	main()
