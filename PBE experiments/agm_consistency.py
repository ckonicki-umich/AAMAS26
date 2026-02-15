import itertools as it
from Node import *
from Infoset import *

def is_conflict(current, new):
	'''
	@arg (tup) node_pair: a tuple object representing a pair of node histories
	@arg (str) current: string representing the binary relation for the input 
		pair currently in the order
	@arg (str) new: string representing a binary relation for the input pair
		that could be used to update the order

	Checks for a conflict between the current binary relation for a given node
	pair in the order and a second binary relation for the same pair that could be
	used to update the order for that pair
	'''
	if current != new:
		return True

	return False

def reversed(node_pair):
	'''
	@arg (tup) node_pair: a tuple object representing a pair of node histories

	Helper method to reverse the order of a pair of node histories, just for the
	sake of being thorough in constructing the order
	'''
	return (node_pair[1], node_pair[0])

def construct_order_given_profile(infoset_list, strategy_profile):
	'''
	@arg (list of Infosets) infoset_list: list of information sets in game tree
	@arg (map: tuple --> (map: str --> float)) strategy_profile: strategy that maps
		each player infoset to a probability distribution over that infoset's corresponding
		action space

	Partially constructs a plausibility order for a game tree given a strategy profile
	'''
	order = {}

	for infoset in infoset_list:
		infoset_strat = strategy_profile.get(infoset.infoset_id)

		for node in infoset.node_list:
			node_history_str = "".join(node.history)
			reachable_nodes = []
			unreachable_nodes = []

			# update order with strategy profile first
			for action in infoset_strat:
				weight = infoset_strat[action]
				pair = (node_history_str, node_history_str + action)

				if weight > 0:
					# h is as plausible as h + action
					order[pair] = "="
					reachable_nodes.append(node_history_str + action)

				else:
					# h is more plausible than h + action
					order[pair] = "<"
					unreachable_nodes.append(node_history_str + action)

			# update plausbility for remainder of information set
			for x in it.product(reachable_nodes, unreachable_nodes):
				order[x] = "<"

	return order

def update_order_given_belief(infoset_list, belief, order):
	'''
	@arg (list of Infosets) infoset_list: list of information sets in game tree
	@arg (dict) belief: map from node histories in the infoset to
		their respective probabilities. Represents that infoset's 
		belief state in this imperfect info game.
	@arg (dict) order: map from pairs of node histories to a binary relation 
		("=", "<", or ">")

	Completes a plausibility order for a game tree given the belief system and current
	order constructed earlier from the strategy profile; halts when any conflicts are
	detected
	'''
	for infoset in infoset_list:
		infoset_belief = belief.get(infoset.infoset_id)
		# make sure the probabilities add up to 1
		assert sum(infoset_belief.values()) == 1.0

		if len(infoset.node_list) > 1:

			possible_nodes = [history for history in infoset_belief if infoset_belief[history] > 0]
			impossible_nodes = [history for history in infoset_belief if history not in possible_nodes]

			for x in it.combinations(possible_nodes, 2):
				y = reversed(x)

				if x in order:
					if is_conflict(order[x], "="):
						return None

					order[x] = "="

				elif y in order:
					if is_conflict(order[y], "="):
						return None

					order[y] = "="

				else:
					order[x] = "="

			for x in it.product(possible_nodes, impossible_nodes):
				y = reversed(x)
				if x in order:
					if is_conflict(order[x], "<"):
						return None

					order[x] = "<"

				elif y in order:
					if is_conflict(order[y], ">"):
						return None

					order[y] = ">"
				else:
					order[x] = "<"

	return order

def is_consistent(infoset_list, strategy_profile, belief):
	'''
	@arg (list of Infosets) infoset_list: list of information sets in game tree
	@arg (map: tuple --> (map: str --> float)) strategy_profile: strategy that maps
		each player infoset to a probability distribution over that infoset's corresponding
		action space
	@arg (dict) belief: map from node histories in the infoset to
		their respective probabilities. Represents that infoset's 
		belief state in this imperfect info game.
	'''

	# First, construct order according to strategy profile
	order = construct_order_given_profile(infoset_list, strategy_profile)

	# Update the order according to the belief system
	order = update_order_given_belief(infoset_list, belief, order)
	if order is None:
		return False

	print("Final Order ", order)
		
	return True


