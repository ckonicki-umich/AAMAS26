import numpy as np
import math
import scipy.stats as stats
import sys
import os
import json
import sys
import re
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
# from matplotlib.ticker import FormatStrFormatter
from collections import Counter
from bargaining import *
from best_response import *

file_ID_list = [
'161GZ',
'BECPD',
'87YP1',
'GTZE3',
'ASNW5'
]

emp_br_list = [
1,
2,
4,
8,
16]

OUTSIDE_OFFERS = ["H", "L"]

NUM_TRIALS = 10
NUM_PLAYER_TURNS = 5


'''
What to look for in the policy maps over time, with different choices of MSS:

- Likelihood of making a deal, given valuations and pool?
- How soon before a deal is made?
- How soon before negotiations end with walking away and taking the outside offer?
- Where are the differences given ne_over_time, spe_over_time, pbe_over_time?
- How valuable are the partitions agreed upon?
- How valuable are the outside offers, if no deal is made?
'''



def gen_output_data_spe(path):
	'''
	@arg (str) game_ID: string prefix identifying game in .npz file name
	@arg (str) br_mss: string (ne or spe) indicating which solution concept we learned to best respond to
		over the course of TE-PSRO
	@arg (str) eval_strat: string(ne or spe) indicating which solution type from the empirical game was
		used to compute regret in the true game
	'''
	try:
		#a_f = np.load(directory + f, allow_pickle=True)
		a_f = np.load(path, allow_pickle=True)
	except:
		print('ha')
		return None

	#print(a_f['arr_0'])
	regret_over_time = a_f['arr_0']
	max_subgame_regret_over_time = a_f['arr_1']
	empirical_game_size_over_time = a_f['arr_2']
	ell_over_time = a_f['arr_3']
	ne_over_time = a_f['arr_4']
	spe_over_time = a_f['arr_5']

	return regret_over_time, max_subgame_regret_over_time, empirical_game_size_over_time, ell_over_time, ne_over_time, spe_over_time


def gen_output_data_pbe(path):
	'''
	@arg (str) game_ID: string prefix identifying game in .npz file name
	@arg (str) br_mss: string (ne or pbe) indicating which solution concept we learned to best respond to
		over the course of TE-PSRO
	@arg (str) eval_strat: string(ne or pbe) indicating which solution type from the empirical game was
		used to compute regret in the true game
	'''
	try:
		#a_f = np.load(directory + f, allow_pickle=True)
		a_f = np.load(path, allow_pickle=True)
	except:
		return None

	regret_over_time = a_f['arr_0']
	max_local_regret_over_time = a_f['arr_1']
	empirical_game_size_over_time = a_f['arr_2']
	ne_over_time = a_f['arr_3']
	pbe_over_time = a_f['arr_4']

	return regret_over_time, max_local_regret_over_time, empirical_game_size_over_time, ne_over_time, pbe_over_time

def gen_nf_output_data(path):
	'''
	@arg (str) game_ID: string prefix identifying game in .npz file name
	@arg (str) br_mss: string (ne or pbe) indicating which solution concept we learned to best respond to
		over the course of TE-PSRO
	@arg (str) eval_strat: string(ne or pbe) indicating which solution type from the empirical game was
		used to compute regret in the true game
	'''
	try:
		# a_f = np.load(directory + f, allow_pickle=True)
		a_f = np.load(path, allow_pickle=True)
	except:
		return None

	regret_over_time = a_f['arr_0']
	empirical_game_size_over_time = a_f['arr_1']
	payoff_err_over_time = a_f['arr_2']
	ne_over_time = a_f['arr_3']

	return regret_over_time, empirical_game_size_over_time, payoff_err_over_time, ne_over_time

def splice_regret_threshold(THRESH, true_regret_over_time):
	regret_copy = []
	for i in range(len(true_regret_over_time)):
		r = true_regret_over_time[i]
		regret_copy.append(r)
		if r <= THRESH:
			return regret_copy

	return regret_copy

def find_subgame_regrets(eval_strat, num_br):
	'''
	'''
	REGRET = []
	for br_mss in ["NE", "PBE"]:
		for i in range(len(file_ID_list)):
			game_ID = file_ID_list[i]
			#print("game ID ", game_ID)
			for trial in range(NUM_TRIALS):

				game_output = gen_output_data(game_ID, br_mss, eval_strat, num_br, trial)
				if game_output is not None:
					max_subgame_regret_over_time = list(game_output[1])

					REGRET.append(max_subgame_regret_over_time)
					#print(max_subgame_regret_over_time)

	#pad = len(max(REGRET, key=len))
	pad = 13
	#print("pad ", pad)
	#REGRET_arr = np.array([x + [x[-1]] * (pad - len(x)) for x in REGRET])
	regret_list = []
	for x in REGRET:
		if len(x) < pad:
			regret_list.append(x + [x[-1]] * (pad - len(x)))
		else:
			regret_list.append(x[:pad])

	REGRET_arr = np.array(regret_list)
	#REGRET_arr = np.array(REGRET)
	x = np.linspace(0, pad - 1, num=pad)
	return x, REGRET_arr

def find_solution_differences(ne, other):
	'''
	@arg (map) ne: NE found in the empirical game tree during a certain iteration of TE-PSRO
	@arg (map) spe: SPE found in the empirical game tree during a certain iteration of TE-PSRO

	Finds the difference(s) between the Nash equilibrium and the Subgame Perfect equilibrium that
	were found for the empirical game of a particular iteration of TE-PSRO
	'''
	diffs = {}
	for k in ne:

		if k not in other:

			diffs[k] = [str(ne[k]), None]

		elif ne[k] != other[k]:

			diffs[k] = [str(ne[k]), str(other[k])]

	for k in other:

		if k not in ne:
			diffs[k] = [None, str(other[k])]

	print("diffs: ne[], other[]")
	for k in diffs:
		print("k ", k, diffs[k])

	return diffs

def retrieve_game(file_ID):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	a_f = np.load("game_parameters.npz", allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		# print(params)
		if "BIG_DoND_" + file_ID == params[0]:
			# print(params)
			return params

def rank_games_by_player_oo_signal(file_ID_list, signal):
	'''
	'''
	ranking_player1 = []
	ranking_player2 = []

	for game_ID in file_ID_list:
		# print(game_ID)
		# print(signal)

		game_params = retrieve_game(game_ID)
		game_param_map = {
			"file_ID": game_params[0],
			"pool": game_params[1], 
			"val_dist": game_params[2],
			"ood1": game_params[3],
			"ood2": game_params[4],
			"o1_pay": game_params[5],
			"o2_pay": game_params[6]
		}

		pool = game_param_map["pool"]
		val_dist = game_param_map["val_dist"]
		outside_offer_dist1 = game_param_map["ood1"]
		outside_offer_dist2 = game_param_map["ood2"]
		o1_pay_arr = game_param_map["o1_pay"]
		o2_pay_arr = game_param_map["o2_pay"]

		#print("o1_pay_arr ", o1_pay_arr)
		#print("o2_pay_arr ", o2_pay_arr)
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, signal)
		#print(o1_pay)
		ranking_player1.append((game_ID, o1_pay))

		o2_pay = get_pay_given_outside_offer(o2_pay_arr, signal)
		#print(o2_pay)
		ranking_player2.append((game_ID, o2_pay))

	ranking_player1.sort(key=lambda x: x[1])
	ranking_player2.sort(key=lambda x: x[1])
	print(ranking_player1)
	print("\n")
	print(ranking_player2)

	return ranking_player1, ranking_player2

def rank_games_by_player_oo_prob(file_ID_list, signal):
	'''
	'''
	ranking_player1 = []
	ranking_player2 = []

	for game_ID in file_ID_list:
		print(game_ID)
		print(signal)

		game_params = retrieve_game(game_ID)
		game_param_map = {
			"file_ID": game_params[0],
			"pool": game_params[1], 
			"val_dist": game_params[2],
			"ood1": game_params[3],
			"ood2": game_params[4],
			"o1_pay": game_params[5],
			"o2_pay": game_params[6]
		}

		pool = game_param_map["pool"]
		val_dist = game_param_map["val_dist"]
		outside_offer_dist1 = game_param_map["ood1"]
		outside_offer_dist2 = game_param_map["ood2"]
		o1_pay_arr = game_param_map["o1_pay"]
		o2_pay_arr = game_param_map["o2_pay"]

		print("outside_offer_dist1 ", outside_offer_dist1)
		print("outside_offer_dist2 ", outside_offer_dist2)

		prob1 = outside_offer_dist1.get(signal)
		print(prob1)
		ranking_player1.append((game_ID, prob1))

		prob2 = outside_offer_dist2.get(signal)
		print(prob2)
		ranking_player2.append((game_ID, prob2))

	ranking_player1.sort(key=lambda x: x[1])
	ranking_player2.sort(key=lambda x: x[1])
	print(ranking_player1)
	print("\n")
	print(ranking_player2)

	return ranking_player1, ranking_player2


def get_negotiation_trajectories(strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
	POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	# list of (action_history, payoff) tuples
	return trajectory_helper([], [], strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

def trajectory_helper(action_history, policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(action_history) - 4
	num_prev_rounds = turns_taken // 2

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		trajectories = []

		for v1 in val_dist.keys():
			next_node = (v1,)
			next_policy_history = (v1,)
			new_trajectories = trajectory_helper(next_node, next_policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
				o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			trajectories = trajectories + new_trajectories

	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		trajectories = []	
		for o1 in OUTSIDE_OFFERS:
			next_node = action_history + (o1,)
			next_policy_history = policy_history + (o1,)
			new_trajectories = trajectory_helper(next_node, next_policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
				o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			trajectories = trajectories + new_trajectories

	# third chance node has been reached -- valuation for player 2 in true game
	elif len(action_history) == 2:
		trajectories = []
		for v2 in val_dist.keys():
			next_node = action_history + (v2,)
			next_policy_history = policy_history + (v2,)
			new_trajectories = trajectory_helper(next_node, next_policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
				o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			trajectories = trajectories + new_trajectories

	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		trajectories = []
		for o2 in OUTSIDE_OFFERS:
			next_node = action_history + (o2,)
			next_policy_history = policy_history + (o2,)
			new_trajectories = trajectory_helper(next_node, next_policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
				o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			trajectories = trajectories + new_trajectories

	elif action_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:

		is_deal = action_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		v1 = action_history[0]
		o1 = action_history[1]
		v2 = action_history[2]
		o2 = action_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
		util_vec = compute_utility(is_deal, pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_prev_rounds)

		traj = (action_history, policy_history, util_vec)
		# print("new traj ")
		# print(action_history)
		# print(policy_history)
		# print(util_vec)
		# print(halt)
		return [traj]

	else:
		trajectories = []
		player_num = 1
		PS = POLICY_SPACE1.copy()
		#BR_network_weights = BR1_weights
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))

		if len(action_history) % 2 == 1:
			player_num = 2
			PS = POLICY_SPACE2.copy()
			#BR_network_weights = BR2_weights

		#print(action_history)
		empir_infoset_id = get_empirical_infoset_id_given_histories(action_history, pool, POLICY_SPACE1, POLICY_SPACE2)
		#print("empir_infoset_id ", empir_infoset_id)

		assert empir_infoset_id[0] == player_num

		infoset_strat = strategy_profile.get(empir_infoset_id)
		game_start = len(action_history) <= 4

		if infoset_strat is not None:
			for empir_action in infoset_strat.keys():
				policy_str = empir_action[0]
				# next_policy_history = policy_history + (empir_action,)
				if infoset_strat.get(empir_action) > 0.0:
					next_policy_history = policy_history + (empir_action,)
					offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
					next_node = action_history + ((offer, empir_action[1]),)

					new_trajectories = trajectory_helper(next_node, next_policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
						o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					trajectories = trajectories + new_trajectories

		else:
			signal = random.choice([True, False])
			
			if len(action_history) <= 5:
				policy_str = "pi_0"
			else:
				last_policy_str = empir_infoset_id[1][-2][0]
				#print("hey")
				if last_policy_str[0] is None:
					last_policy_str = "pi_0"

				policy_str = last_policy_str

			empir_action = (policy_str, signal)
			next_policy_history = policy_history + (empir_action,)
			offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
			a = (offer, signal)
			next_node = action_history + (a,)

			new_trajectories = trajectory_helper(next_node, next_policy_history, strategy_profile, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
				o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			trajectories = trajectories + new_trajectories

	return trajectories


def get_deal_frequency(game_ID, solution_i, policy_map1, policy_map2):
	'''
	'''
	print("get_deal_frequency")
	game_params = retrieve_game(game_ID)
	game_param_map = {
		"file_ID": game_params[0],
		"pool": game_params[1], 
		"val_dist": game_params[2],
		"ood1": game_params[3],
		"ood2": game_params[4],
		"o1_pay": game_params[5],
		"o2_pay": game_params[6]
	}

	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	deal_freq_spe_list = []
	deal_freq_ne_list = []

	deal_freq_spe = {}
	deal_freq_ne = {}

	# Use compute_true_pay to track history and payoffs
	trajectories_i = get_negotiation_trajectories(solution_i, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr,
		policy_map1, policy_map2)

	print(len(trajectories_i))

	for t in trajectories_i:
		action_history, policy_history, pay = t
		j = len(action_history) % 2
		num_turns_taken = len(action_history) - 4
		# increment count for how likely deal was to be made
		if action_history[-1][0] == ("deal",):
			deal_freq_spe[(j, num_turns_taken, "deal")] = deal_freq_spe.get((j, num_turns_taken, "deal"), 0) + 1

		# increment count for how likely to walk away
		elif action_history[-1][0] == ("walk",):
			deal_freq_spe[(j, num_turns_taken, "walk")] = deal_freq_spe.get((j, num_turns_taken, "walk"), 0) + 1

		# increment count for being out of time
		else:
			deal_freq_spe[("OOT")] = deal_freq_spe.get(("OOT"), 0) + 1

	print(deal_freq_spe)
	deal_freq_spe_list.append(deal_freq_spe)
	#print(donny)

	#print(no)
	return deal_freq_spe_list

def compute_relative_pay_for_player(action_history_t, pool, o1_pay_arr, o2_pay_arr, player_j):
	'''
	'''
	v1 = action_history_t[0]
	#print("v1 ", v1)
	o1 = action_history_t[1]
	#print("o1 ", o1)
	v2 = action_history_t[2]
	#print("v2 ", v2)
	o2 = action_history_t[3]
	#print("o2 ", o2)
	turns_taken = len(action_history_t) - 4
	num_prev_rounds = turns_taken // 2
	offer = action_history_t[-1][0]
	#print("offer ", offer)
	o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
	o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
	oo_pay_j = compute_utility(("walk",), pool, v1, v2, offer, o1_pay, o2_pay, num_prev_rounds)[player_j]
	# print("o1_pay ", o1_pay)
	# print("o2_pay ", o2_pay)
	#print("oo_pay_j ", oo_pay_j)

	offer_pay_j = compute_utility(("deal",), pool, v1, v2, offer, o1_pay, o2_pay, num_prev_rounds)[player_j]
	#print("offer_pay_j ", offer_pay_j)

	#print(halt)
	return offer_pay_j - oo_pay_j


def analyze_trajectories(game_ID, trajectories_i, policy_map1, policy_map2):
	'''
	'''
	#print("analyze_trajectories")
	game_params = retrieve_game(game_ID)
	game_param_map = {
		"file_ID": game_params[0],
		"pool": game_params[1], 
		"val_dist": game_params[2],
		"ood1": game_params[3],
		"ood2": game_params[4],
		"o1_pay": game_params[5],
		"o2_pay": game_params[6]
	}

	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	pay_traj_deal_list = []
	pay_traj_walk_list = []

	pay_traj1_deal = []
	pay_traj1_walk = []
	pay_traj2_deal = []
	pay_traj2_walk = []

	# trajectories_i = get_negotiation_trajectories(solution_i, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr,
	# 	policy_map1, policy_map2)

	for k in range(len(trajectories_i)):
	#for k in range(15):
		#print(k)
		t = trajectories_i[k]
		action_history, policy_history, pay = t

		pay_traj1_i = []
		pay_traj2_i = []

		for i in range(4, len(action_history)):
			a = action_history[i]
			#print('a ', a)
			if a[0] in [("deal",), ("walk",)]:
				break
			p1 = compute_relative_pay_for_player(action_history[:i+1], pool, o1_pay_arr, o2_pay_arr, 0)
			p2 = compute_relative_pay_for_player(action_history[:i+1], pool, o1_pay_arr, o2_pay_arr, 1)

			pay_traj1_i.append(p1)
			pay_traj2_i.append(p2)

		#print(pay_traj1)
		#print(pay_traj2)
		#pay_traj1.append(pay_traj1_i)
		#pay_traj2.append(pay_traj2_i)
		if action_history[-1][0] == ("deal",):
			pay_traj1_deal.append(pay_traj1_i)
			pay_traj2_deal.append(pay_traj2_i)
		else:
			pay_traj1_walk.append(pay_traj1_i)
			pay_traj2_walk.append(pay_traj2_i)

	#negotiations_spe_list.append([pay_traj1, pay_traj2])
	pay_traj_deal_list.append([pay_traj1_deal, pay_traj2_deal])
	pay_traj_walk_list.append([pay_traj1_walk, pay_traj2_walk])
	

	return pay_traj_deal_list, pay_traj_walk_list

def get_repeat_freq(player_offers):
	'''
	'''
	count = dict(Counter(player_offers))
	repeat = float(max(count.values())) / len(player_offers)

	return repeat

def check_for_repeats(game_ID, solution_i, policy_map1, policy_map2):
	'''
	'''
	print("check_for_repeats")
	game_params = retrieve_game(game_ID)
	game_param_map = {
		"file_ID": game_params[0],
		"pool": game_params[1], 
		"val_dist": game_params[2],
		"ood1": game_params[3],
		"ood2": game_params[4],
		"o1_pay": game_params[5],
		"o2_pay": game_params[6]
	}

	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	repeats_list = []

	repeat_freq1 = []
	repeat_freq2 = []

	trajectories_i = get_negotiation_trajectories(solution_i, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr,
		policy_map1, policy_map2)

	for i in range(len(trajectories_i)):
		t = trajectories_i[i]
		action_history, policy_history, pay = t

		#print(len(action_history))
		#print(policy_history)

		player1_offers = [action_history[i][0] for i in range(4, len(action_history), 2)]
		#print(player1_offers)
		player2_offers = [action_history[i][0] for i in range(5, len(action_history), 2)]
		#print(player2_offers)

		repeat1 = None
		repeat2 = None

		if len(player1_offers) == 0:
			repeat1 = 0.0

		else:
			repeat1 = get_repeat_freq(player1_offers)

		if len(player2_offers) == 0:
			repeat2 = 0.0
		else:
			repeat2 = get_repeat_freq(player2_offers)

		repeat_freq1.append(repeat1)
		repeat_freq2.append(repeat2)

		#print(stahp)
	# print(repeat_freq1)
	# print(repeat_freq2)

	repeats_list.append([repeat_freq1, repeat_freq2])

	return repeats_list

def get_offer_values_by_p1(game_ID, trajectories_i, policy_map1, policy_map2):
	'''
	'''
	game_params = retrieve_game(game_ID)
	game_param_map = {
		"file_ID": game_params[0],
		"pool": game_params[1], 
		"val_dist": game_params[2],
		"ood1": game_params[3],
		"ood2": game_params[4],
		"o1_pay": game_params[5],
		"o2_pay": game_params[6]
	}

	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	offer_val_list_deal = []
	offer_val_list_walk = []

	offer_vals1_deal = []
	offer_vals1_walk = []
	offer_vals2_deal = []
	offer_vals2_walk = []

	# trajectories_i = get_negotiation_trajectories(solution_i, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr,
	# 	policy_map1, policy_map2)

	#for i in range(len(trajectories_i)):
	for i in range(15):
		t = trajectories_i[i]
		action_history, policy_history, pay = t
		#print('action_history ', action_history)

		v1 = action_history[0]
		#print("v1 ", v1)
		o1 = action_history[1]
		#print("o1 ", o1)
		v2 = action_history[2]
		#print("v2 ", v2)
		o2 = action_history[3]
		#print("o2 ", o2)
		# turns_taken = len(action_history) - 4
		# num_prev_rounds = turns_taken // 2

		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		all_offers_i = [action_history[i][0] for i in range(4, len(action_history))]

		offer_vals1_i = []
		offer_vals2_i = []

		first_offer = all_offers_i[0]
		print("first_offer ", first_offer)
		first_offer_pay = compute_utility(("deal",), pool, v1, v2, first_offer, o1_pay, o2_pay, 0)
		first_offer_pay_1 = first_offer_pay[0]
		first_offer_pay_2 = first_offer_pay[1]
		print(first_offer_pay_1, first_offer_pay_2)

		for k in range(1, len(all_offers_i)):
			offer = all_offers_i[k]
			#print(offer)
			if offer in [("deal",), ("walk",)]:
				break

			turns_taken = k + 1
			num_prev_rounds = turns_taken // 2

			offer_pay = compute_utility(("deal",), pool, v1, v2, offer, o1_pay, o2_pay, num_prev_rounds)
			print(offer_pay)
			'''
			if first_offer_pay_1 == 0.0:
				offer_vals1_i.append(offer_pay[0] / 0.1)
			else:
				print(offer_pay[0] / first_offer_pay_1)
				offer_vals1_i.append(offer_pay[0] / first_offer_pay_1)
			
			if first_offer_pay_2 == 0.0:
				offer_vals2_i.append(offer_pay[1] / 0.1)
			else:
				print(offer_pay[1] / first_offer_pay_2)
				offer_vals2_i.append(offer_pay[1] / first_offer_pay_2)
			'''
			offer_vals1_i.append(offer_pay[0] - first_offer_pay_1)
			offer_vals2_i.append(offer_pay[1] - first_offer_pay_2)

		# print(offer_vals1_i)
		# print(offer_vals2_i)
		

		# offer_vals1.append(offer_vals1_i)
		# offer_vals2.append(offer_vals2_i)
		if action_history[-1][0] == ("deal",):
			offer_vals1_deal.append(offer_vals1_i)
			offer_vals2_deal.append(offer_vals2_i)
		else:
			offer_vals1_walk.append(offer_vals1_i)
			offer_vals2_walk.append(offer_vals2_i)

	#offer_val_list.append([offer_vals1, offer_vals2])
	offer_val_list_walk.append([offer_vals1_walk, offer_vals2_walk])
	offer_val_list_deal.append([offer_vals1_deal, offer_vals2_deal])
	#return offer_val_list
	return offer_val_list_deal, offer_val_list_walk

# '''
# # SPE for MSS, EVAL
# DATA_SPE = []
# DATA_NE = []
# DATA_PBE = []

# for num_br in emp_br_list:
# 	#print(num_br)
# 	#'''
# 	directory = "res_spe_eval_spe_mss_BR" + str(num_br) + "/"
# 	for trial in range(5):
# 		for game_ID in file_ID_list:
# 			#print(num_br, trial, game_ID)
# 			output_f = "BIG_DoND_SPE_BR_SPE_EVAL_NUM_EMPIR_BR" + str(num_br) + "_BIG_DoND_" + str(game_ID) + "_" + str(trial) + ".npz"

# 			policy1_f = "NUM_EMPIR_BR" + str(num_br) + "_" + str(trial) + "_BIG_DoND_" + game_ID + "_SPE_mss_SPE_eval_policy_map1.npy"
# 			policy2_f = "NUM_EMPIR_BR" + str(num_br) + "_" + str(trial) + "_BIG_DoND_" + game_ID + "_SPE_mss_SPE_eval_policy_map2.npy"


# 			try:
# 				a_output_f = np.load(directory + output_f, allow_pickle=True)
# 				a_policy1_f = np.load(directory + policy1_f, allow_pickle=True)
# 				a_policy2_f = np.load(directory + policy2_f, allow_pickle=True)
# 			except:
# 				#print("no")
# 				continue

# 			#print(a_output_f['arr_1'])
# 			true_regret_over_time = splice_regret_threshold(0.1, a_output_f['arr_0'])
# 			spe_over_time = a_output_f['arr_5']
# 			policy_map1 = a_policy1_f.item()
# 			policy_map2 = a_policy2_f.item()

# 			# use max_len
# 			if true_regret_over_time[-1] < 0.1 and len(true_regret_over_time) < 13:

# 				DATA_SPE.append((game_ID, num_br, spe_over_time, policy_map1, policy_map2, true_regret_over_time))
# 			#print(a_policy1_f.item())

# 	directory = "res_spe_eval_ne_mss_BR" + str(num_br) + "/"
# 	for trial in range(5):
# 		for game_ID in file_ID_list:
# 			output_f = "BIG_DoND_NE_BR_SPE_EVAL_NUM_EMPIR_BR" + str(num_br) + "_BIG_DoND_" + str(game_ID) + "_" + str(trial) + ".npz"

# 			policy1_f = "NUM_EMPIR_BR" + str(num_br) + "_" + str(trial) + "_BIG_DoND_" + game_ID + "_NE_mss_SPE_eval_policy_map1.npy"
# 			policy2_f = "NUM_EMPIR_BR" + str(num_br) + "_" + str(trial) + "_BIG_DoND_" + game_ID + "_NE_mss_SPE_eval_policy_map2.npy"				

# 			try:
# 				a_output_f = np.load(directory + output_f, allow_pickle=True)
# 				a_policy1_f = np.load(directory + policy1_f, allow_pickle=True)
# 				a_policy2_f = np.load(directory + policy2_f, allow_pickle=True)
# 			except:
# 				#print("no")
# 				continue

# 			#print(a_output_f['arr_1'])
# 			true_regret_over_time = splice_regret_threshold(0.1, a_output_f['arr_0'])
# 			ne_over_time = a_output_f['arr_4']
# 			policy_map1 = a_policy1_f.item()
# 			policy_map2 = a_policy2_f.item()

# 			# use max_len
# 			if true_regret_over_time[-1] < 0.1 and len(true_regret_over_time) < 13:

# 				DATA_NE.append((game_ID, num_br, ne_over_time, policy_map1, policy_map2, true_regret_over_time))
# 			#print(a_policy1_f.item())
# 	#'''

# 	#'''
# 	# TODO 9/12: RUN WHEN DATA IS ACQUIRED FROM JOBS
# 	directory = "res_ne_eval_pbe_mss_BR" + str(num_br) + "/"
# 	for trial in range(5):
# 		for game_ID in file_ID_list:
# 			output_f = "BIG_DoND_PBE_BR_NE_EVAL_NUM_EMPIR_BR" + str(num_br) + "_BIG_DoND_" + str(game_ID) + "_" + str(trial) + ".npz"

# 			policy1_f = "NUM_EMPIR_BR" + str(num_br) + "_" + str(trial) + "_BIG_DoND_" + game_ID + "_PBE_mss_NE_eval_policy_map1.npy"
# 			policy2_f = "NUM_EMPIR_BR" + str(num_br) + "_" + str(trial) + "_BIG_DoND_" + game_ID + "_PBE_mss_NE_eval_policy_map2.npy"				

# 			try:
# 				a_output_f = np.load(directory + output_f, allow_pickle=True)
# 				a_policy1_f = np.load(directory + policy1_f, allow_pickle=True)
# 				a_policy2_f = np.load(directory + policy2_f, allow_pickle=True)
# 			except:
# 				#print("no")
# 				continue

# 			#print(a_output_f['arr_1'])
# 			true_regret_over_time = splice_regret_threshold(0.1, a_output_f['arr_0'])
# 			pbe_over_time = a_output_f['arr_4']
# 			policy_map1 = a_policy1_f.item()
# 			policy_map2 = a_policy2_f.item()

# 			# use max_len
# 			if true_regret_over_time[-1] < 0.1 and len(true_regret_over_time) < 13:

# 				DATA_PBE.append((game_ID, num_br, pbe_over_time, policy_map1, policy_map2, true_regret_over_time))
# 			#print(a_policy1_f.item())
	#'''

# TODO 5/15: For each possible true game state, how likely is it that a deal will be made, and when?
# How does this change over time?
# How does the nature of the counteroffers over time change, both over the course of negotiations and
# over the course of TE-PSRO's run looking at the returned solutions?

# What does the trajectory of counteroffers mean with respect to the game parameters/offer space? 
# How do they escalate or change the dynamic of negotiations?
# - repeats instead of switching to a new offer?
# - going back to an old counteroffer from earlier?
# - compare projected pay of accepting current proposal vs. projected pay of walking + taking outside offer?

# ADD CODE TO LOOK AT OFFER SPACE AND WHAT'S MORE LIKELY TO BE COUNTEROFFERED, AND HOW MUCH PAYOFF IT
# COULD YIELD FOR EACH PLAYER


# NOTE THE STRINGS ARE ACTUALLY BACKWARDS: "H" denotes the low offer and "L" denotes the high offer!

game_ranking_by_H = rank_games_by_player_oo_signal(file_ID_list, "H")
game_ranking_by_L = rank_games_by_player_oo_signal(file_ID_list, "L")

game_ranking_by_H_prob = rank_games_by_player_oo_prob(file_ID_list, "H")
game_ranking_by_L_prob = rank_games_by_player_oo_prob(file_ID_list, "L")


# np.savez_compressed("spe_data_912", DATA_SPE)
# np.savez_compressed("ne_data_912", DATA_NE)
# np.savez_compressed("pbe_data_914", DATA_PBE)

data_files = [
"ne_data_912.npz",
"spe_data_912.npz",
"pbe_data_914.npz"
]

trajectory_files = [
"trajectories_ne_data_919.npz",
"trajectories_spe_data_919.npz",
"trajectories_pbe_data_919.npz"
]

offer_val_files = [
"offer_vals_ne_data_919.npz",
"offer_vals_spe_data_919.npz",
"offer_vals_pbe_data_919.npz"
]

MSS = [
"ne",
"spe",
"pbe"
]

mss_id = int(sys.argv[1])
DATA = np.load(data_files[mss_id], allow_pickle=True)['arr_0']
OFFER_VALS = []
TRAJECTORIES = []

# INDEX OVER EACH OF THESE ARRAYS BY CORR. INDEX IN DATA_SPE!!!
print(len(DATA))
for i in range(len(DATA)):
#for i in range(5):
	print(i)
	data = DATA[i]
	game_id = data[0]
	#print(game_id)
	num_br = data[1]
	spe_over_time = data[2]
	policy_map1 = data[3]
	policy_map2 = data[4]
	true_regret_over_time = data[5]
	#print(true_regret_over_time)

	game_params = retrieve_game(game_id)
	game_param_map = {
		"file_ID": game_params[0],
		"pool": game_params[1], 
		"val_dist": game_params[2],
		"ood1": game_params[3],
		"ood2": game_params[4],
		"o1_pay": game_params[5],
		"o2_pay": game_params[6]
	}

	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	soln = spe_over_time[-1]
	if mss_id == 2:
		soln = spe_over_time[-1][0]

	trajectories_i = get_negotiation_trajectories(soln, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr,
		policy_map1, policy_map2)
	
	# x1 = get_deal_frequency(game_id, spe_over_time[-1], policy_map1, policy_map2)
	# DEAL_FREQ_SPE.append(x1)
		
	# x2 = analyze_trajectories(game_id, soln, policy_map1, policy_map2)
	# TRAJECTORIES.append(x2)
	'''
	x2 = analyze_trajectories(game_id, trajectories_i, policy_map1, policy_map2)
	TRAJECTORIES.append(x2)
	'''
	
	# x3 = check_for_repeats(game_id, spe_over_time[-1], policy_map1, policy_map2)
	# REPEATS_SPE.append(x3)
	# #print(true_regret_over_time)
	
	# x4 = get_offer_values_by_p1(game_id, soln, policy_map1, policy_map2)
	# OFFER_VALS.append(x4)
	x4 = get_offer_values_by_p1(game_id, trajectories_i, policy_map1, policy_map2)
	OFFER_VALS.append(x4)
	

# np.savez_compressed("deal_freq_spe_data_912", DEAL_FREQ_SPE)
np.savez_compressed(trajectory_files[mss_id], TRAJECTORIES)
# np.savez_compressed("repeats_spe_data_912", REPEATS_SPE)
np.savez_compressed(offer_val_files[mss_id], OFFER_VALS)

