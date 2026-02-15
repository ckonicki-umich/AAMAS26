from bargaining import *
from compute_pay import *

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

def get_total_nf_budget(SAMPLING_BUDGET, complete_psro_iter):
	'''
	'''
	num_cells_square = complete_psro_iter**2
	num_new_cells_square = (complete_psro_iter + 1)**2

	return (num_new_cells_square - num_cells_square) * SAMPLING_BUDGET

def simulate(val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, pool, old_strategy_space, BR, total_NF_sample_budget, noise, payoffs, 
	POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
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
	num_iter = int(float(total_NF_sample_budget) / num_tree_paths)
	
	for strategy in generate_new_BR_paths(old_strategy_space, BR):

		for n in range(num_iter):

			v1, v2 = generate_player_valuations(val_dist)
			o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)
			o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
			o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
			action_history = [v1, o1, v2, o2]
			empir_history = [o1, o2]
			utility = None
			reached_end = False

			for turn in range(NUM_PLAYER_TURNS):
				empir_infoset_id1 = get_empirical_infoset_id_given_histories(tuple(action_history), pool, POLICY_SPACE1, POLICY_SPACE2)
				num_rounds = (len(action_history) - 4) // 2

				# player 1 makes initial offer to player 2
				offer_space1 = generate_offer_space(pool)
				action_space = list(it.product(offer_space1, [True, False]))

				# don't allow the action taken or the policy chosen to advise player 1 to accept a deal if nothing has happened
				if turn == 0:
					offer_space1.remove(('deal',))
					offer_space1.remove(('walk',))

				p1_empir_action = None
				if empir_infoset_id1 not in strategy:
					p1_empir_action = empir_history[-2]
					p1_empir_action = (p1_empir_action[0], bool(p1_empir_action[1]))
					if not reached_end:
						reached_end = True
						empir_history += [p1_empir_action]
				else:
					p1_empir_action = random.choice(strategy.get(empir_infoset_id1))
					p1_empir_action = (p1_empir_action[0], bool(p1_empir_action[1]))
					empir_history += [p1_empir_action]

				p1_policy_str = p1_empir_action[0]
				p1_offer = get_offer_given_policy(p1_policy_str, action_space, POLICY_SPACE1, tuple(action_history), pool)
				p1_action = (p1_offer, p1_empir_action[1])
				action_history += [p1_action]
				
                # check for walking or deal
				if p1_action[0] in [("walk",), ("deal",)]:
					empir_history += [p1_empir_action]

					if p1_action[0] == ("walk",):
						utility = compute_utility(("walk",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)
					else:
						utility = compute_utility(("deal",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)
					break

				# player 2 makes counteroffer to player 1
				empir_infoset_id2 = get_empirical_infoset_id_given_histories(tuple(action_history), pool, POLICY_SPACE1, POLICY_SPACE2)
				num_rounds = (len(action_history) - 4) // 2
				offer_space2 = generate_offer_space(pool)
				action_space = list(it.product(offer_space2, [True, False]))

				p2_empir_action = None
				if empir_infoset_id2 not in strategy:
					if len(empir_history) > 4:
						p2_empir_action = empir_history[-2]
					else:
						p2_empir_action = default_policy2

					p2_empir_action = (p2_empir_action[0], bool(p2_empir_action[1]))
					if not reached_end:
						reached_end = True
						empir_history += [p2_empir_action]
				else:
					p2_empir_action = random.choice(strategy.get(empir_infoset_id2))
					p2_empir_action = (p2_empir_action[0], bool(p2_empir_action[1]))
					empir_history += [p2_empir_action]
				
				p2_offer = get_offer_given_policy(p2_empir_action[0], action_space, POLICY_SPACE2, tuple(action_history), pool)
				p2_action = (p2_offer, p2_empir_action[1])
				action_history += [p2_action]

				# check for walking or deal
				if p2_action[0] in [("walk",), ("deal",)]:
					if p2_action[0] == ("walk",):
						utility = compute_utility(("walk",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)
					else:				
						utility = compute_utility(("deal",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)

					break

			observations[tuple(empir_history)] = observations.get(tuple(empir_history), 0.0) + 1
			
            # ran out of time to make a deal/end negotiations --> number of rounds elapsed needs to equal
			# the number of player turns
			if utility is None:
				utility = compute_utility(("walk",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, NUM_PLAYER_TURNS)

			payoff_sample = np.random.normal(utility, np.array([noise] * 2))
			payoffs[tuple(empir_history)] = payoffs.get(tuple(empir_history), []) + [payoff_sample]

	return payoffs, observations
