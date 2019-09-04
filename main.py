from base import train, predict
from node import Node
from functions import sigmoid

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
	n = Node(3)
	n.connect(lambda p: print(f"label: {p}"))
	print(f"*Before training*")
	n.transmit_all(features)
	n.train(goal_pred, features)
	print(f"*After training*")
	n.transmit_all(features)

def triple_neuron():
	goal_pred = 10.0
	features = [1.0, 3.0, 2.0]
	def done(p):
		print(f'pred: {p}\ndelta: error: {p-goal_pred}\n{(p-goal_pred)**2}')
	print(f"goal: {goal_pred}")
	n1 = Node(3, sigmoid)
	n2 = Node(1, sigmoid)
	n3 = Node(1, sigmoid)
	n1.connect(lambda p: n2.transmit(0, p))
	n2.connect(lambda p: n3.transmit(0, p))
	n3.connect(done)
	print(f"*Before training*")
	n1.transmit_all(features)

	#n1.train(goal_pred, features)
	#print(f"*After training*")
	#n1.transmit_all(features)


def main():
	#simple_case()
	#neuron()
	triple_neuron()


if __name__ == '__main__':
	main()