from best_response import *
from compute_pay import *

def compute_regret(meta_strategy, eval_string, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, file_ID, hp_set1, hp_set2, 
	POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (map) ms: given strategy profile
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (str) file_ID: identification string for file containing outputs (error, regret, etc.)
	@arg (map) hp_set1: learned hyperparameters for player 1's DQN for best response
	@arg (map) hp_set2: learned hyperparameters for player 2's DQN for best response

	Computes regret for both players for a given strategy profile and returns the higher of the two regrets
	'''
	meta_strategy_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_empirical_strategy_pay(meta_strategy, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

	print("true ms pay ", meta_strategy_pay)
	print("computing regret")
	print(eval_string)
	regrets = []

	BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(meta_strategy, eval_string, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
		o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, 1000, True)

	for j in range(2):
		action_pay = None
		if j == 0:
			action_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_pay(meta_strategy, BR1_weights, BR2_weights, 1, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			
		else:
			action_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_pay(meta_strategy, BR1_weights, BR2_weights, 2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

		regrets.append(max(action_pay[j] - meta_strategy_pay[j], 0.0))

	print("max regrets ", max(regrets))
	return max(regrets)