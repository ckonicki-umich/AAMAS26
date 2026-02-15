import random
import itertools as it
from best_response import *

def generate_new_BR_paths(empir_strat_space, BR):
	'''
	'''
	NUM_BR = 2**len(BR)

	# get every possible combination of BR's to be included in tree
	for j in range(1, NUM_BR):
		strat = {}
		bin_list = [int(x) for x in bin(j)[2:]]
		
		if len(bin_list) < len(BR):
			bin_list_copy = bin_list[:]
			bin_list = [0] * (len(BR) - len(bin_list_copy)) + bin_list_copy

		br_list = list(it.compress(BR, bin_list))
		for infoset_id in empir_strat_space:
			if infoset_id in br_list:
				strat[infoset_id] = [BR[infoset_id]]
			else:
				strat[infoset_id] = empir_strat_space[infoset_id][:]

		yield strat

	if NUM_BR == 1:
		strat = {}
		for infoset_id in empir_strat_space:
			strat[infoset_id] = empir_strat_space[infoset_id][:]

		yield strat

def get_total_nf_budget(SAMPLING_BUDGET, complete_psro_iter):
	'''
	'''
	num_cells_square = complete_psro_iter**2
	#print("num_cells_square ", num_cells_square)
	num_new_cells_square = (complete_psro_iter + 1)**2
	#print("num_new_cells_square ", num_new_cells_square)

	return (num_new_cells_square - num_cells_square) * SAMPLING_BUDGET


def simulate(game_param_map, old_strategy_space, BR, total_NF_sample_budget, noise, payoffs, POLICY_SPACE1, POLICY_SPACE2, 
	default_policy1, default_policy2):
	'''
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (map: Infoset --> (map: str --> float)) strategy_profile: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (int) num_iter: number of iterations
	@arg (float) noise: Gaussian noise added to the utility samples outputted at the end
		of a single simulation
	@arg (dict) payoffs: dictionary mapping histories to sampled utilities

	Black-box simulator that will be used for EGTA. Simulates a single run through the
	true game, num_iter times. Returns the observations and utilities returned for each run.

	Note: Simulates pure strategies ONLY
	'''
	observations = {}
	num_tree_paths = max(2**len(BR) - 1, 1)
	print("num_tree_paths ", num_tree_paths)
	num_iter = int(float(total_NF_sample_budget) / num_tree_paths)
	print("num_iter ", num_iter)

	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	for strategy in generate_new_BR_paths(old_strategy_space, BR):
		for n in range(num_iter):
			action_history = []
			empir_history = []
			policy_history = []

			for r in range(num_rounds - 1):
				outcome, event_index = sample_stochastic_event_given_history(tuple(action_history), chance_events, card_weights)
				action_history += [outcome]
				policy_history += [outcome]
				empir_history += [outcome]
				empir_infoset_id1 = get_empirical_infoset_id_given_empir_history(policy_history, 1)
				p1_empir_action = None

				if empir_infoset_id1 not in strategy:
					p1_empir_action = empir_history[-2]
					empir_history += [p1_empir_action]
				else:
					p1_empir_action = random.choice(strategy.get(empir_infoset_id1))
					empir_history += [p1_empir_action]

				policy_history += [p1_empir_action]
				p1_action = get_action_given_policy(p1_empir_action, p1_actions, POLICY_SPACE1, tuple(action_history), game_params)
				action_history += [p1_action]
				empir_infoset_id2 = get_empirical_infoset_id_given_empir_history(policy_history, 2)
				
				p2_empir_action = None
				if empir_infoset_id2 not in strategy:
					if r == 0:
						p2_empir_action = default_policy2
					else:
						p2_empir_action = empir_history[-3]

					empir_history += [p2_empir_action]

				else:
					p2_empir_action = random.choice(strategy.get(empir_infoset_id2))
					empir_history += [p2_empir_action]

				policy_history += [p2_empir_action]
				p2_action = get_action_given_policy(p2_empir_action, p2_actions, POLICY_SPACE1, tuple(action_history), game_params)
				action_history += [p2_action]

			utility = get_utility(action_history, num_rounds, payoff_map)
			observations[tuple(empir_history)] = observations.get(tuple(empir_history), 0.0) + 1
			payoff_sample = np.random.normal(utility, np.array([noise] * 2))
			payoffs[tuple(empir_history)] = payoffs.get(tuple(empir_history), []) + [payoff_sample]

	return payoffs, observations