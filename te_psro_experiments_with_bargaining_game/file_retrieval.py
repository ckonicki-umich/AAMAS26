import json
import numpy as np
import itertools as it

def retrieve_game(file_ID_index):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	a_f = np.load("game_parameters.npz", allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		if file_ID_index == params[0]:
			return params

def retrieve_json_hps(file_ID_index, player_num):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration
	@arg (int) player_num: index {1, 2} corresponding to one of the two players

	Helper method to retrieve each set of learned hyperparameter values from
	phases 1 and 2 for a given game and player
	'''
	with open('optimal_learned_hp_DoND.json') as f:
		data = f.read()
	js = json.loads(data)

	d_both = js[str(file_ID_index)]
	d = d_both[str(player_num)]

	keys = d.keys()
	hp_set = None
	for elt in it.product(*d.values()):

		hp_set = dict(zip(keys, elt))
	
	return hp_set