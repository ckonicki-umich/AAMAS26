import json
import numpy as np
import itertools as it


def retrieve_game(file_ID_index, num_rounds):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	print("num_rounds ", num_rounds)
	file_name = "game_parameters_" + str(num_rounds) + "_rounds.npz"
	print("file_name ", file_name)
	a_f = np.load(file_name, allow_pickle=True)
	params = a_f['arr_0'][file_ID_index]

	return params

def retrieve_json_hps(num_rounds, player_num):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration
	@arg (int) player_num: index {1, 2} corresponding to one of the two players

	Helper method to retrieve each set of learned hyperparameter values from
	phases 1 and 2 for a given game and player
	'''
	with open('PBE experiments/best_response_computation/optimal_learned_hp_abstract.json') as f:
		data = f.read()

	js = json.loads(data)

	d_both = js[str(num_rounds)]
	d = d_both[str(player_num)]

	keys = d.keys()
	hp_set = None
	for elt in it.product(*d.values()):
		hp_set = dict(zip(keys, elt))

	return hp_set