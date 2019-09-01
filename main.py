from base import train, predict
from neuron import Neuron

def simple_case():
	features = [1.0, 3.0, 2.0]
	weights = [0, 0, 0]
	goal_pred = 10.0
	alpha = 1e-2
	times = 100
	print(f"goal_prediction={goal_pred}")
	label = predict(features, weights)
	print(f"*Before training*")
	print(f"prediction={label}")
	print(f"*After training*")
	weights = train(features, weights, goal_pred, times, alpha)
	label = predict(features, weights)
	print(f"prediction={label}")
	print(f"confidence={round(100-(abs(label-goal_pred))*100/(label+goal_pred), 2)}%")
	print(f"weights={weights}")


def neuron():
	goal_pred = 10.0
	features = [1.0, 3.0, 2.0]
	print(f"goal_prediction={goal_pred}")
	n = Neuron(3, lambda x: x if x > 0 else 0)
	n.connect(lambda p: print(f"label: {p}"))
	print(f"*Before training*")
	n.transmit_all(features)
	n.train(goal_pred, features)
	print(f"*After training*")
	n.transmit_all(features)


def main():
	simple_case()
	neuron()


if __name__ == '__main__':
	main()