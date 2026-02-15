import numpy as np
np.float = np.float64
import sys
import mmappickle as mmp
from Node import *
from Infoset import *
from ExtensiveForm import *
from DQN import *
from compute_memory import *
from best_response import *
from compute_pay import *
from file_retrieval import *
from initial_policy import *
from simulation_code import *
from config import Config

config = Config.load_config("pbe_algorithm_config.yaml")

NUM_PLAYERS = config.num_players
SAMPLING_BUDGET = config.sampling_budget
STD_DEV = config.std_dev
NUM_PSRO_ITER = config.num_psro_iter
T_list = config.T_list
emp_br_list = config.emp_br_list
num_rounds_list = config.num_rounds_list

HANDLERS = {
	ExtensiveForm: ExtensiveFormHandler,
	Infoset: InfosetHandler,
	Node: NodeHandler
}

def main(game_param_map, trial_index):
	'''
	@arg (map) initial_sigma: Initial metastrategy based on empirical strategy
		space
	@arg (map) game_param_map: map of game parameters for given file ID (player valuation distribution, item pool,
		outside offer distributions)
	@arg (int) T: Number of iterations for a single run of CFR (whether solving a game for NE or a subgame as part
		of solving a game for SPE)
	@arg (str) br_mss: identification for which solution type we will use as the MSS to find best responses
		for -- at present, either "ne" or "spe"
	@arg (str) eval_strat: identification for which solution type we will use as the strategy against which to
		compute true game regret and worst-case subgame regret in the empirical game -- at present either "ne"
		or "spe"

	Runs a single play of TE-EGTA on large abstract game, expanding strategy space, simulating
	each new strategy and constructing the empirical game model, which is then solved for an
	approximate NE using counter-factual regret minimization (CFR) and approximate PBE using PBE-CFR in
	order to compare correctness and scalability
	'''
	file_ID = game_param_map["file_ID"]
	num_rounds = game_param_map["num_rounds"]
	included_rounds = game_param_map["included_rounds"]

	file_ID = file_ID + "_" + str(included_rounds[-1])
	print(file_ID)

	#extract learned hyperparameters for DQN
	hp_set1 = retrieve_json_hps(num_rounds, 1)
	hp_set2 = retrieve_json_hps(num_rounds, 2)

	empir_root = Node(0, (0, 1), [], [], NUM_PLAYERS)
	X = {}

	initial_sigma, default_policy1, default_policy2, POLICY_SPACE1, POLICY_SPACE2 = construct_initial_policy(game_param_map, hp_set1, hp_set2)

	# Initialize the empirical strategy space based on initial_sigma
	empir_strat_space = {}
	for i in initial_sigma.keys():
		empir_strat_space[i] = [list(initial_sigma[i].keys())[0]]

	prefix = 'NUM_EMPIR_BR' + str(NUM_EMPIR_BR) + '_' + str(trial_index) + '_' + file_ID + '_PBE_alg_test'
	file_name = prefix + '_empirical_game.mmdpickle'
	print("file_name ", file_name)

	with open(file_name, 'w') as fp:
		pass
	mgame = mmp.mmapdict(file_name)

	empirical_game = ExtensiveForm([[], []], empir_root, [], {}, num_rounds)
	mgame['game'] = empirical_game
	max_infoset_regret_over_time = []
	ne_over_time = []
	pbe_over_time = []
	runtimes = []

	payoffs = {}
	br_meta_strat = initial_sigma.copy()
	empirical_game_size_over_time = []

	BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(br_meta_strat, "ne_mss", prefix, game_param_map, hp_set1, hp_set2, 
		POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, False)

	player1_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 1]
	num_br_samples = min(NUM_EMPIR_BR, len(player1_empir_infostates))
	policy_str = "pi_" + str(len(POLICY_SPACE1))
	POLICY_SPACE1[policy_str] = BR1_weights
	infoset_gains1 = []

	for infoset_id in player1_empir_infostates:
		infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 1, BR1_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
		if infoset_freq * infoset_gain > 0:
			infoset_gains1.append(infoset_gain * infoset_freq)
		else:
			infoset_gains1.append(config.null_infosets)

	x = np.arange(len(player1_empir_infostates))
	infoset_inds_1 = None
	try:
		infoset_inds_1 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains1))
	except:
		num_nonzero = len([y for y in softmax(infoset_gains1) if y > 0.0])
		infoset_inds_1 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains1))
	
	player1_empir_M = [player1_empir_infostates[i] for i in infoset_inds_1]
	BR1 = convert_into_best_response_policy(player1_empir_M, policy_str, BR1_weights, game_param_map)
	print("BR1 ", BR1)
	
	infoset_gains2 = []
	player2_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 2]
	num_br_samples = min(NUM_EMPIR_BR, len(player2_empir_infostates))
	policy_str = "pi_" + str(len(POLICY_SPACE2))
	POLICY_SPACE2[policy_str] = BR2_weights

	for infoset_id in player2_empir_infostates:
		infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 2, BR2_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
		if infoset_freq * infoset_gain > 0:
			infoset_gains2.append(infoset_gain * infoset_freq)
		else:
			infoset_gains2.append(config.null_infosets)

	x = np.arange(len(player2_empir_infostates))
	infoset_inds_2 = None
	try:
		infoset_inds_2 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains2))
	except:
		num_nonzero = len([y for y in softmax(infoset_gains2) if y > 0.0])
		infoset_inds_2 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains2))	

	player2_empir_M = [player2_empir_infostates[i] for i in infoset_inds_2]
	BR2 = convert_into_best_response_policy(player2_empir_M, policy_str, BR2_weights, game_param_map)

	BR = {}
	BR.update(BR1)
	BR.update(BR2)

	empirical_game_size_over_time.append([len(empir_strat_space), total_size(empirical_game, HANDLERS)])
	print("empirical_game_size_over_time so far ", empirical_game_size_over_time)

	# save policy maps to disk
	np.save(prefix + "_policy_map1.npy", POLICY_SPACE1)
	np.save(prefix + "_policy_map2.npy", POLICY_SPACE2)

	while len(empirical_game_size_over_time) < NUM_PSRO_ITER:
		
		old_empir_strat_space = empir_strat_space.copy()

		# Load policy maps from disk
		POLICY_SPACE1 = np.load(prefix + "_policy_map1.npy", allow_pickle=True).item()
		POLICY_SPACE2 = np.load(prefix + "_policy_map2.npy", allow_pickle=True).item()

		print("simulating true game and updating empirical game w/ new simulation data")

		total_NF_sample_budget = get_total_nf_budget(SAMPLING_BUDGET, len(empirical_game_size_over_time))

		if len(empirical_game_size_over_time) == 1:
			payoffs, new_observations = simulate(game_param_map, empir_strat_space, {}, SAMPLING_BUDGET, STD_DEV, payoffs, POLICY_SPACE1, POLICY_SPACE2,
				default_policy1, default_policy2)
			empirical_game = mgame['game']

			empirical_game.update_game_with_simulation_output(new_observations, payoffs)
			mgame['game'] = empirical_game
		
		payoffs, new_observations = simulate(game_param_map, empir_strat_space, BR, total_NF_sample_budget, STD_DEV, payoffs, POLICY_SPACE1, POLICY_SPACE2,
			default_policy1, default_policy2)
		empirical_game = mgame['game']
		empirical_game.update_game_with_simulation_output(new_observations, payoffs)
		mgame['game'] = empirical_game
		del new_observations

		for i in range(2):
			for infoset in empirical_game.infosets[i]:
				infoset_id = infoset.infoset_id
				if infoset_id not in old_empir_strat_space.keys():
					empir_strat_space[infoset_id] = infoset.action_space[:]

				else:
					empir_strat_space[infoset_id] = infoset.action_space[:]

		empirical_game_size_over_time.append([len(empir_strat_space), total_size(empirical_game, HANDLERS)])
		print("empirical_game_size_over_time so far ", empirical_game_size_over_time)

		print("computing new metastrategy")
		ne_ms = None
		pbe_ms = None

		time_list = []
		regret_list = []
		for T in [500, 1000, 2000, 5000]:
			print("T ", T)
			start_CFR = time.process_time()
			ne_ms = empirical_game.cfr(T)
			time_CFR = time.process_time() - start_CFR
			print("time_CFR ", time_CFR)
			ne_over_time.append(ne_ms)
			start_PBE = time.process_time()
			pbe_ms = empirical_game.compute_PBE(T)
			time_PBE = time.process_time() - start_PBE
			print("time_PBE ", time_PBE)
			pbe_over_time.append(pbe_ms)
			time_list.append([str(T), time_PBE, time_CFR])
			pbe_infoset_regrets = empirical_game.verify_sequent_rat(pbe_ms)

			regret_list.append([str(T), pbe_infoset_regrets])
			print("regret_list ", regret_list)

		runtimes.append(np.array([len(empir_strat_space), time_list], dtype=object))
		max_infoset_regret_over_time.append(np.array(regret_list))

		br_meta_strat = ne_ms.copy()
		BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(br_meta_strat, "ne_mss", prefix, game_param_map, hp_set1, hp_set2, 
			POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, False)

		player1_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 1]
		num_br_samples = min(NUM_EMPIR_BR, len(player1_empir_infostates))
		policy_str = "pi_" + str(len(POLICY_SPACE1))
		POLICY_SPACE1[policy_str] = BR1_weights

		infoset_gains1 = []
		for infoset_id in player1_empir_infostates:
			infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 1, BR1_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			if infoset_freq * infoset_gain > 0:
				infoset_gains1.append(infoset_gain * infoset_freq)
			else:
				infoset_gains1.append(config.null_infosets)

		x = np.arange(len(player1_empir_infostates))
		infoset_inds_1 = None
		try:
			infoset_inds_1 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains1))
		except:
			num_nonzero = len([y for y in softmax(infoset_gains1) if y > 0.0])
			infoset_inds_1 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains1))
			
		player1_empir_M = [player1_empir_infostates[i] for i in infoset_inds_1]
		BR1 = convert_into_best_response_policy(player1_empir_M, policy_str, BR1_weights, game_param_map)

		infoset_gains2 = []
		player2_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 2]
		num_br_samples = min(NUM_EMPIR_BR, len(player2_empir_infostates))
		policy_str = "pi_" + str(len(POLICY_SPACE2))
		POLICY_SPACE2[policy_str] = BR2_weights

		for infoset_id in player2_empir_infostates:
			infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 2, BR2_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			if infoset_freq * infoset_gain > 0:
				infoset_gains2.append(infoset_gain * infoset_freq)
			else:
				infoset_gains2.append(config.null_infosets)

		x = np.arange(len(player2_empir_infostates))
		infoset_inds_2 = None
		try:
			infoset_inds_2 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains2))
		except:
			num_nonzero = len([y for y in softmax(infoset_gains2) if y > 0.0])
			infoset_inds_2 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains2))

		player2_empir_M = [player2_empir_infostates[i] for i in infoset_inds_2]
		BR2 = convert_into_best_response_policy(player2_empir_M, policy_str, BR2_weights, game_param_map)

		BR = {}
		BR.update(BR1)
		BR.update(BR2)

		# save policy maps to disk
		np.save(prefix + "_policy_map1.npy", POLICY_SPACE1)
		np.save(prefix + "_policy_map2.npy", POLICY_SPACE2)

		np.savez_compressed(prefix, max_infoset_regret_over_time, empirical_game_size_over_time, ne_over_time, pbe_over_time, runtimes)

	del mgame['game']
	mgame.vacuum()

'''
MAIN CODE
'''

file_ID_index = int(sys.argv[1]) // 9
print("file_ID_index ", file_ID_index)

trial_index = int(sys.argv[1]) % 3
print("trial_index ", trial_index)

br_index = int(sys.argv[2])
print("br_index ", br_index)
NUM_EMPIR_BR = emp_br_list[br_index]
print("NUM_EMPIR_BR ", NUM_EMPIR_BR)

rounds_index = int(sys.argv[3])
print("rounds_index ", rounds_index)
NUM_ROUNDS = num_rounds_list[rounds_index]

included_rounds_index = int(sys.argv[1]) // 3
print("included_rounds_index ", included_rounds_index)
included_rounds = [i for i in range(included_rounds_index + 1)]

game_params = retrieve_game(file_ID_index, NUM_ROUNDS)

game_param_map = {
	"file_ID": game_params[0],
	"num_rounds": game_params[1],
	"p1_actions": game_params[2],
	"p2_actions": game_params[3],
	"chance_events": game_params[4],
	"card_weights": game_params[5],
	"payoff_map": game_params[6],
	"included_rounds": included_rounds
}

for x in game_param_map:
	print(x, game_param_map[x])

main(game_param_map, trial_index)



