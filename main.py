def predict(
	inputs:list,
	weights:list
)->float: # pred
	"""
		Make a prediction
	"""
	assert type(inputs) == list
	assert type(weights) == list
	assert len(inputs) == len(weights)
	return sum([inputs[k]*weights[k] for k in range(len(inputs))])


def train(
	inputs:list,
	weights:list,
	goal_pred:float,
	times:int=20,
	alpha:float=1.0
)->list: # updated weights
	""" Transform weights to reach the goal prediction """
	assert type(inputs) == list
	assert type(weights) == list
	assert len(inputs) == len(weights)
	assert type(goal_pred) == float
	assert type(times) == int
	assert type(alpha) == float
	assert alpha != 0
	for _ in range(times):
		pred = predict(inputs, weights)
		delta = pred-goal_pred
		weight_deltas = [inputs[k]*delta for k in range(len(inputs))]
		weights = [weights[k]-(weight_deltas[k]*alpha) for k in range(len(inputs))]
	return weights

def main():
	inputs = [1.0, 3.0, 2.0]
	weights = [0.5, 2.0, 0.1]
	goal_pred = 10.0
	alpha = 1e-2
	times = 10
	print(f"goal_prediction={goal_pred}")
	pred = predict(
		inputs,
		weights
	)
	print(f"*Before training*")
	print(f"prediction={pred}")
	print(f"*After training*")
	weights = train(
		inputs,
		weights,
		goal_pred,
		times,
		alpha
	)
	pred = predict(
		inputs,
		weights
	)
	print(f"prediction={pred}")
	print(f"confidence={round(100-(abs(pred-goal_pred))*100/(pred+goal_pred), 2)}%")
	print(f"weights={weights}")


if __name__ == '__main__':
	main()
