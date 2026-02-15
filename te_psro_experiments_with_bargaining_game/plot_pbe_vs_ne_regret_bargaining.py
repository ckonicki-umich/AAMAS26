import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

ALPHA = 0.2
COLOR1 = "#228B22"
COLOR2 = "#C3438F"
COLOR3 = "#0060AD"
COLOR4 = "#191970"
COLOR5 = "#A018CE"
COLOR6 = "#5E718B"

plt.rcParams.update({
    "text.usetex": True, 
    "font.family": "serif",
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def gen_output_data(path):
	'''
	@arg (str) game_ID: string prefix identifying game in .npz file name
	@arg (str) br_mss: string (ne or spe) indicating which solution concept we learned to best respond to
		over the course of TE-PSRO
	@arg (str) eval_strat: string(ne or spe) indicating which solution type from the empirical game was
		used to compute regret in the true game
	'''
	try:
		a_f = np.load(path, allow_pickle=True)
	except:
		return None

	regret_over_time = a_f['arr_0']
	max_local_regret_over_time = a_f['arr_1']
	empirical_game_size_over_time = a_f['arr_2']
	ne_over_time = a_f['arr_3']
	pbe_over_time = a_f['arr_4']

	return regret_over_time, max_local_regret_over_time, empirical_game_size_over_time, ne_over_time, pbe_over_time

def splice_regret_threshold(THRESH, true_regret_over_time):
	regret_copy = []
	for i in range(len(true_regret_over_time)):
		r = true_regret_over_time[i]
		regret_copy.append(r)
		if r <= THRESH:
			return regret_copy

	return regret_copy

def get_true_regret_arr_over_time(directory, threshold=0.0):
	'''
	'''
	REGRET = []
	path_list = Path(directory).glob('*.npz')
	for path in path_list:
		game_output = gen_output_data(str(path))

		if game_output is not None:
			true_regret_over_time = splice_regret_threshold(threshold, game_output[0])
			REGRET.append(list(true_regret_over_time))
				
	pad = len(max(REGRET, key=len))
	REGRET_arr = np.array([x + [x[-1]] * (pad - len(x)) for x in REGRET])
	x = np.linspace(0, pad - 1, num=pad)

	return x, REGRET_arr

def gen_two_plots(res_directory_ne_as_mss, res_directory_pbe_as_mss, num_trials):
	'''
	Docstring for gen_two_plots
	
	:param res_directory_ne_as_mss: Description
	:param res_directory_pbe_as_mss: Description
	'''
	x_ne_mss, REGRET_NE_MSS_arr = get_true_regret_arr_over_time(res_directory_ne_as_mss)
	x_pbe_mss, REGRET_PBE_MSS_arr = get_true_regret_arr_over_time(res_directory_pbe_as_mss)

	avg_regret_ne = np.mean(REGRET_NE_MSS_arr, axis=0)
	std_regret_ne = np.std(REGRET_NE_MSS_arr, axis=0)
	avg_regret_pbe = np.mean(REGRET_PBE_MSS_arr, axis=0)
	std_regret_pbe = np.std(REGRET_PBE_MSS_arr, axis=0)

	# ASSUME SAME NUMBER OF DATA SAMPLES IN EACH DIRECTORY
	CONF_VAL = 1.95 / num_trials**0.5

	plt.figure(1, (6, 4))
	plt.plot(x_ne_mss, avg_regret_ne, color=COLOR1, label="MSS = NE")
	plt.fill_between(x_ne_mss, avg_regret_ne - std_regret_ne * CONF_VAL, avg_regret_ne + std_regret_ne * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR1)
	plt.plot(x_pbe_mss, avg_regret_pbe, color=COLOR2, label="MSS = PBE")
	plt.fill_between(x_pbe_mss, avg_regret_pbe - std_regret_pbe * CONF_VAL, avg_regret_pbe + std_regret_pbe * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR2)

	plt.yticks(weight='bold')
	plt.xticks(np.arange(0, max(len(x_ne_mss), len(x_pbe_mss)), step=5), weight='bold')
	plt.xlabel(r"\textbf{TE-PSRO Epochs}")
	plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
	plt.legend(loc='upper right')
	plt.show()

########################################################################################################################################################################
# PLOTTING TRUE GAME REGRET -- 2 plots by MSS = {PBE, NE}, EVAL = NE
########################################################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("dir_ne_as_mss")
parser.add_argument("dir_pbe_as_mss")
parser.add_argument("num_trials")
args = parser.parse_args()

res_directory_ne_as_mss = args.dir_ne_as_mss
res_directory_pbe_as_mss = args.dir_pbe_as_mss
num_trials = int(args.num_trials)

gen_two_plots(res_directory_ne_as_mss=res_directory_ne_as_mss, res_directory_pbe_as_mss=res_directory_pbe_as_mss, num_trials=num_trials)

