import statistics
import numpy as np
import sys



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



def retrieve_game(file_ID):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	a_f = np.load("game_parameters.npz", allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		if "BIG_DoND_" + file_ID == params[0]:
			return params


def get_relative_pay_map(DATA, RELATIVE_PAY_TRAJECTORIES):
	'''
	'''
	RELATIVE_PAY1_DEAL_MAP = {}
	RELATIVE_PAY1_WALK_MAP = {}
	RELATIVE_PAY2_DEAL_MAP = {}
	RELATIVE_PAY2_WALK_MAP = {}

	for k in range(len(DATA)):
		game_ID = DATA[k][0]
		pay_trajs_deal, pay_trajs_walk = RELATIVE_PAY_TRAJECTORIES[k]
		
		pay_trajs1_deal = pay_trajs_deal[0][0]
		pay_trajs2_deal = pay_trajs_deal[0][1]
		pay_trajs1_walk = pay_trajs_walk[0][0]
		pay_trajs2_walk = pay_trajs_walk[0][1]

		for x in range(len(pay_trajs1_deal)):
			pay1 = pay_trajs1_deal[x]
			pay2 = pay_trajs2_deal[x]
			num_turns = len(pay1)
			RELATIVE_PAY1_DEAL_MAP[(game_ID, num_turns)] = RELATIVE_PAY1_DEAL_MAP.get((game_ID, num_turns), []) + [pay1]
			RELATIVE_PAY2_DEAL_MAP[(game_ID, num_turns)] = RELATIVE_PAY2_DEAL_MAP.get((game_ID, num_turns), []) + [pay2]

		for x in range(len(pay_trajs1_walk)):
			pay1 = pay_trajs1_walk[x]
			pay2 = pay_trajs2_walk[x]
			num_turns = len(pay1)
			RELATIVE_PAY1_WALK_MAP[(game_ID, num_turns)] = RELATIVE_PAY1_WALK_MAP.get((game_ID, num_turns), []) + [pay1]
			RELATIVE_PAY2_WALK_MAP[(game_ID, num_turns)] = RELATIVE_PAY2_WALK_MAP.get((game_ID, num_turns), []) + [pay2]			

	return RELATIVE_PAY1_DEAL_MAP, RELATIVE_PAY2_DEAL_MAP, RELATIVE_PAY1_WALK_MAP, RELATIVE_PAY2_WALK_MAP

def get_mean_relative_pay(RELATIVE_PAY1_MAP, RELATIVE_PAY2_MAP, sorted_key_list):
	'''
	'''
	MEAN_RELATIVE_PAY1 = {}
	MEAN_RELATIVE_PAY2 = {}

	for game_ID in file_ID_list:
		for k in sorted_key_list:

			relative_pay1_list = None
			relative_pay2_list = None
			try:
				relative_pay1_list = RELATIVE_PAY1_MAP[(game_ID, k)]
				relative_pay2_list = RELATIVE_PAY2_MAP[(game_ID, k)]
			except:
				continue

			mean_pay1 = np.mean(relative_pay1_list, axis=0)
			mean_pay2 = np.mean(relative_pay2_list, axis=0)
			
			MEAN_RELATIVE_PAY1[(game_ID, k)] = mean_pay1
			MEAN_RELATIVE_PAY2[(game_ID, k)] = mean_pay2

	return MEAN_RELATIVE_PAY1, MEAN_RELATIVE_PAY2


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

MSS = [
"ne",
"spe",
"pbe"
]

mss_id = int(sys.argv[1])
DATA = np.load(data_files[mss_id], allow_pickle=True)['arr_0']
TRAJECTORIES = np.load(trajectory_files[mss_id], allow_pickle=True)['arr_0']

RELATIVE_PAY1_DEAL_MAP, RELATIVE_PAY2_DEAL_MAP, RELATIVE_PAY1_WALK_MAP, RELATIVE_PAY2_WALK_MAP = get_relative_pay_map(DATA, TRAJECTORIES)
key_list_walk = list(set([x[1] for x in RELATIVE_PAY1_WALK_MAP.keys()]))
# print(key_list_walk)

key_list_deal = list(set([x[1] for x in RELATIVE_PAY1_DEAL_MAP.keys()]))
#print(key_list_deal)

sorted_key_list_deal = sorted(key_list_deal)
sorted_key_list_walk = sorted(key_list_walk)

MEAN_RELATIVE_PAY1_DEAL, MEAN_RELATIVE_PAY2_DEAL = get_mean_relative_pay(RELATIVE_PAY1_DEAL_MAP, RELATIVE_PAY2_DEAL_MAP, sorted_key_list_deal)
MEAN_RELATIVE_PAY1_WALK, MEAN_RELATIVE_PAY2_WALK = get_mean_relative_pay(RELATIVE_PAY1_WALK_MAP, RELATIVE_PAY2_WALK_MAP, sorted_key_list_walk)
# MEAN_RELATIVE_PAY1, MEAN_RELATIVE_PAY2 = get_mean_relative_pay(RELATIVE_PAY1_MAP, RELATIVE_PAY2_MAP, sorted_key_list)
# for k in MEAN_RELATIVE_PAY1:
# 	print(k, MEAN_RELATIVE_PAY1[k])
# print("\n")

# for k in MEAN_RELATIVE_PAY2:
# 	print(k, MEAN_RELATIVE_PAY2[k])
# print(stop)

np.savez_compressed("mean_relative_pay1_" + MSS[mss_id], sorted_key_list_deal, sorted_key_list_walk, MEAN_RELATIVE_PAY1_DEAL, MEAN_RELATIVE_PAY1_WALK)
np.savez_compressed("mean_relative_pay2_" + MSS[mss_id], sorted_key_list_deal, sorted_key_list_walk, MEAN_RELATIVE_PAY2_DEAL, MEAN_RELATIVE_PAY2_WALK)







