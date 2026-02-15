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

def gen_six_plots(ne_as_mss_directories, pbe_as_mss_directories, num_trials):
	'''
	Docstring for gen_six_plots
	
	:param ne_as_mss_directories: Description
	:param pbe_as_mss_directories: Description
	:param num_trials: Description
	'''
	dir_ne_as_mss_ir0, dir_ne_as_mss_ir1, dir_ne_as_mss_ir2 = ne_as_mss_directories
	dir_pbe_as_mss_ir0, dir_pbe_as_mss_ir1, dir_pbe_as_mss_ir2 = pbe_as_mss_directories

	x_ne_mss_ir0, REGRET_NE_MSS_arr_ir0 = get_true_regret_arr_over_time(dir_ne_as_mss_ir0)
	x_ne_mss_ir1, REGRET_NE_MSS_arr_ir1 = get_true_regret_arr_over_time(dir_ne_as_mss_ir1)
	x_ne_mss_ir2, REGRET_NE_MSS_arr_ir2 = get_true_regret_arr_over_time(dir_ne_as_mss_ir2)

	x_pbe_mss_ir0, REGRET_PBE_MSS_arr_ir0 = get_true_regret_arr_over_time(dir_pbe_as_mss_ir0)
	x_pbe_mss_ir1, REGRET_PBE_MSS_arr_ir1 = get_true_regret_arr_over_time(dir_pbe_as_mss_ir1)
	x_pbe_mss_ir2, REGRET_PBE_MSS_arr_ir2 = get_true_regret_arr_over_time(dir_pbe_as_mss_ir2)

	avg_regret_ne_ir0 = np.mean(REGRET_NE_MSS_arr_ir0, axis=0)
	std_regret_ne_ir0 = np.std(REGRET_NE_MSS_arr_ir0, axis=0)
	avg_regret_ne_ir1 = np.mean(REGRET_NE_MSS_arr_ir1, axis=0)
	std_regret_ne_ir1 = np.std(REGRET_NE_MSS_arr_ir1, axis=0)
	avg_regret_ne_ir2 = np.mean(REGRET_NE_MSS_arr_ir2, axis=0)
	std_regret_ne_ir2 = np.std(REGRET_NE_MSS_arr_ir2, axis=0)

	avg_regret_pbe_ir0 = np.mean(REGRET_PBE_MSS_arr_ir0, axis=0)
	std_regret_pbe_ir0 = np.std(REGRET_PBE_MSS_arr_ir0, axis=0)
	avg_regret_pbe_ir1 = np.mean(REGRET_PBE_MSS_arr_ir1, axis=0)
	std_regret_pbe_ir1 = np.std(REGRET_PBE_MSS_arr_ir1, axis=0)
	avg_regret_pbe_ir2 = np.mean(REGRET_PBE_MSS_arr_ir2, axis=0)
	std_regret_pbe_ir2 = np.std(REGRET_PBE_MSS_arr_ir2, axis=0)

	# ASSUME SAME NUMBER OF DATA SAMPLES IN EACH DIRECTORY
	CONF_VAL = 1.95 / num_trials**0.5

	plt.figure(1, (6, 4))
	plt.plot(x_ne_mss_ir0, avg_regret_ne_ir0, color=COLOR1, label="MSS = NE, IR = [0]")
	plt.fill_between(x_ne_mss_ir0, avg_regret_ne_ir0 - std_regret_ne_ir0 * CONF_VAL, avg_regret_ne_ir0 + std_regret_ne_ir0 * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR1)
	plt.plot(x_ne_mss_ir1, avg_regret_ne_ir1, color=COLOR2, label="MSS = NE, IR = [0, 1]")
	plt.fill_between(x_ne_mss_ir1, avg_regret_ne_ir1 - std_regret_ne_ir1 * CONF_VAL, avg_regret_ne_ir1 + std_regret_ne_ir1 * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR2)
	plt.plot(x_ne_mss_ir2, avg_regret_ne_ir2, color=COLOR3, label="MSS = NE, IR = [0, 1, 2]")
	plt.fill_between(x_ne_mss_ir2, avg_regret_ne_ir2 - std_regret_ne_ir2 * CONF_VAL, avg_regret_ne_ir2 + std_regret_ne_ir2 * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR3)
	plt.plot(x_pbe_mss_ir0, avg_regret_pbe_ir0, color=COLOR4, label="MSS = PBE")
	plt.fill_between(x_pbe_mss_ir0, avg_regret_pbe_ir0 - std_regret_pbe_ir0 * CONF_VAL, avg_regret_pbe_ir0 + std_regret_pbe_ir0 * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR4)
	plt.plot(x_pbe_mss_ir1, avg_regret_pbe_ir1, color=COLOR5, label="MSS = PBE, IR = [0, 1]")
	plt.fill_between(x_pbe_mss_ir1, avg_regret_pbe_ir1 - std_regret_pbe_ir1 * CONF_VAL, avg_regret_pbe_ir1 + std_regret_pbe_ir1 * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR5)
	plt.plot(x_pbe_mss_ir2, avg_regret_pbe_ir2, color=COLOR6, label="MSS = PBE, IR = [0, 1, 2]")
	plt.fill_between(x_pbe_mss_ir2, avg_regret_pbe_ir2 - std_regret_pbe_ir2 * CONF_VAL, avg_regret_pbe_ir2 + std_regret_pbe_ir2 * CONF_VAL,
		alpha=ALPHA, facecolor=COLOR6)

	plt.yticks(weight='bold')
	plt.xticks(np.arange(0, max(len(x_ne_mss_ir0), len(x_ne_mss_ir1), len(x_ne_mss_ir2), len(x_pbe_mss_ir0), len(x_pbe_mss_ir1), 
							 len(x_pbe_mss_ir2)), step=5), weight='bold')
	plt.xlabel(r"\textbf{TE-PSRO Epochs}")
	plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
	plt.legend(loc='upper right')
	plt.show()

########################################################################################################################################################################
# PLOTTING TRUE GAME REGRET -- 6 plots by MSS = {PBE, NE}, EVAL = NE, INCLUDED_ROUNDS = 1st (0), 1st and 2nd (0, 1), or all (0, 1, 2)
########################################################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("dir_ne_as_mss_ir0")
parser.add_argument("dir_ne_as_mss_ir1")
parser.add_argument("dir_ne_as_mss_ir2")
parser.add_argument("dir_pbe_as_mss_ir0")
parser.add_argument("dir_pbe_as_mss_ir1")
parser.add_argument("dir_pbe_as_mss_ir2")
parser.add_argument("num_trials")
args = parser.parse_args()

res_directory_ne_as_mss = (args.dir_ne_as_mss_ir0, args.dir_ne_as_mss_ir1, args.dir_ne_as_mss_ir2)
res_directory_pbe_as_mss = (args.dir_pbe_as_mss_ir0, args.dir_pbe_as_mss_ir1, args.dir_pbe_as_mss_ir2)
num_trials = int(args.num_trials)

gen_six_plots(res_directory_ne_as_mss, res_directory_pbe_as_mss, num_trials)

