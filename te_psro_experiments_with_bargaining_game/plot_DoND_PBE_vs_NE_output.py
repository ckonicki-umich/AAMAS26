import numpy as np
import math
import scipy.stats as stats
import sys
import os
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from brokenaxes import brokenaxes

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

OFFER_SIGNALS = ["H", "L"]

NUM_TRIALS = 10

def gen_output_data(game_ID, br_mss, eval_strat, num_empir_br, trial):
	'''
	@arg (str) game_ID: string prefix identifying game in .npz file name
	@arg (str) br_mss: string (ne or pbe) indicating which solution concept we learned to best respond to
		over the course of TE-PSRO
	@arg (str) eval_strat: string(ne or pbe) indicating which solution type from the empirical game was
		used to compute regret in the true game
	'''
	#print("br_mss ", br_mss, "eval_strat ", eval_strat)

	#directory = "res_" + eval_strat + "_eval_" + br_mss + "_mss/"
	#directory = "res_30_" + eval_strat + "_eval_" + br_mss + "_mss/"
	directory = "res_" + eval_strat + "_eval_" + br_mss + "_mss_BR" + str(num_empir_br) + "/"
	#f = "BIG_DoND_" + br_mss + "_BR_" + eval_strat + "_EVAL_T" + str(500) + "_" + "BIG_DoND_" + game_ID + ".npz"
	f = "BIG_DoND_" + br_mss.upper() + "_BR_" + eval_strat.upper() + "_EVAL_NUM_EMPIR_BR" + str(num_empir_br)
	f = f + "_BIG_DoND_" + game_ID + "_" + str(trial) + ".npz"
	#print("f ", f)
	#print("directory ", directory)
	try:
		a_f = np.load(directory + f, allow_pickle=True)
	except:
		return None

	regret_over_time = a_f['arr_0']
	max_local_regret_over_time = a_f['arr_1']
	empirical_game_size_over_time = a_f['arr_2']
	ne_over_time = a_f['arr_3']
	pbe_over_time = a_f['arr_4']

	#print("empirical_game_size_over_time ", empirical_game_size_over_time)
	#print(ell_over_time)
	#print(max_subgame_regret_over_time)

	#print("regret_over_time ", regret_over_time)

	return regret_over_time, max_local_regret_over_time, empirical_game_size_over_time, ne_over_time, pbe_over_time

def gen_nf_output_data(game_ID, trial):
	'''
	@arg (str) game_ID: string prefix identifying game in .npz file name
	@arg (str) br_mss: string (ne or pbe) indicating which solution concept we learned to best respond to
		over the course of TE-PSRO
	@arg (str) eval_strat: string(ne or pbe) indicating which solution type from the empirical game was
		used to compute regret in the true game
	'''
	# directory = "res_nf_psro_30/"
	directory = "res_nf_psro_6_16/"
	#f = "BIG_DoND_" + br_mss + "_BR_" + eval_strat + "_EVAL_T" + str(500) + "_" + "BIG_DoND_" + game_ID + ".npz"
	f = "BIG_DoND_NF-PSRO_BIG_DoND_" + game_ID + "_" + str(trial) + ".npz"
	#print("directory ", directory)
	try:
		a_f = np.load(directory + f, allow_pickle=True)
	except:
		return None

	regret_over_time = a_f['arr_0']
	empirical_game_size_over_time = a_f['arr_1']
	payoff_err_over_time = a_f['arr_2']
	ne_over_time = a_f['arr_3']

	return regret_over_time, empirical_game_size_over_time, payoff_err_over_time, ne_over_time

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

def is_perfect_information_state(infoset_id):
	'''
	'''
	state_history = infoset_id[1]
	num_signals_present = len([h for h in state_history if h in OFFER_SIGNALS])

	if num_signals_present == 2:
		return True

	return False

def splice_regret_threshold(THRESH, true_regret_over_time):
	regret_copy = []
	for i in range(len(true_regret_over_time)):
		r = true_regret_over_time[i]
		regret_copy.append(r)
		if r <= THRESH:
			return regret_copy

	return regret_copy


def plot_true_regret_over_time(eval_strat, br_mss, num_br, max_len):
	'''
	'''
	REGRET = []

	for i in range(len(file_ID_list)):
		game_ID = file_ID_list[i]
		#print("game ID ", game_ID, "eval:", eval_strat, " mss:", br_mss)
		for trial in range(NUM_TRIALS):
			game_output = gen_output_data(game_ID, br_mss, eval_strat, num_br, trial)
			if game_output is not None:
				true_regret_over_time = splice_regret_threshold(0.1, game_output[0])
				
				#if len(true_regret_over_time) > 1 and len(true_regret_over_time) < 17:
				#'''
				if true_regret_over_time[-1] < 0.1 and len(true_regret_over_time) < max_len:
					#print(list(true_regret_over_time))
					REGRET.append(list(true_regret_over_time))
				#'''
				#REGRET.append(list(true_regret_over_time))
				
				# if br_mss == "pbe":
				# 	if true_regret_over_time[-1] < 1.0:
				# 		print(list(true_regret_over_time))
				# 		REGRET.append(list(true_regret_over_time))
				# else:
				# 	if true_regret_over_time[-1] < 2.0:
				# 		print(list(true_regret_over_time))
				# 		REGRET.append(list(true_regret_over_time))
				

	pad = len(max(REGRET, key=len))
	#print("pad ", pad)
	REGRET_arr = np.array([x + [x[-1]] * (pad - len(x)) for x in REGRET])
	x = np.linspace(0, pad - 1, num=pad)

	return x, REGRET_arr

def plot_nf_true_regret_over_time():
	'''
	'''
	REGRET = []

	for i in range(len(file_ID_list)):
		game_ID = file_ID_list[i]
		#print("game ID ", game_ID, "eval:", eval_strat, " mss:", br_mss)
		for trial in range(NUM_TRIALS):
			game_output = gen_nf_output_data(game_ID, trial)
			if game_output is not None:
				true_regret_over_time = game_output[0]
				#print(list(true_regret_over_time))
				if len(true_regret_over_time) > 1 and true_regret_over_time[-1] < 1.0:
					'''
					if num_br == 2:
						print(list(true_regret_over_time))
					if br_mss == "pbe":
						if true_regret_over_time[-1] < 1.0:
							REGRET.append(list(true_regret_over_time))
					else:
						if true_regret_over_time[-1] < 3.0:
							REGRET.append(list(true_regret_over_time))
					'''
					print(list(true_regret_over_time))
					REGRET.append(list(true_regret_over_time))

	pad = len(max(REGRET, key=len))
	print("pad ", pad)
	REGRET_arr = np.array([x + [x[-1]] * (pad - len(x)) for x in REGRET])
	x = np.linspace(0, pad - 1, num=pad)
	#print(no)

	return x, REGRET_arr

def plot_empirical_game_size_over_time(num_br):
	#print("num_br ", num_br)
	SIZE = []
	MEM = []

	for eval_strat in ["NE", "PBE"]:
		for br_mss in ["NE", "PBE"]:

			for i in range(len(file_ID_list)):
				game_ID = file_ID_list[i]
				#print("game ID ", game_ID, "eval:", eval_strat, " mss:", br_mss)
				for trial in range(NUM_TRIALS):
					game_output = gen_output_data(game_ID, br_mss, eval_strat, num_br, trial)
					if game_output is not None:
						empirical_game_size_over_time = game_output[2]

						#print("empirical_game_size_over_time in num_infosets and GB")
						#print(empirical_game_size_over_time)
						num_infosets = list(empirical_game_size_over_time[:, 0])

						# converting into MB
						memory_req = list([x * 1000.0 for x in empirical_game_size_over_time[:, 1]])

						#print("num_infosets ", num_infosets)
						#print("memory_req ", memory_req)

						SIZE.append(num_infosets)
						MEM.append(memory_req)

			# print("SIZE ", SIZE)
			#print(len(SIZE))

	#pad_size = len(max(SIZE, key=len))
	pad_size = 13
	#print("pad_size ", pad_size)
	# for x in SIZE:
	# 	print()
	size_list = []
	for x in SIZE:
		if len(x) < pad_size:
			size_list.append(x + [x[-1]] * (pad_size - len(x)))
		else:
			size_list.append(x[:pad_size])
	# SIZE_arr = np.array([x + [x[-1]] * (pad_size - len(x)) for x in SIZE])
	SIZE_arr = np.array(size_list)
	#SIZE_arr = np.array(SIZE)

	#pad_mem = len(max(MEM, key=len))
	pad_mem = 13
	#print("pad_mem ", pad_mem)
	mem_list = []
	for x in MEM:
		if len(x) < pad_mem:
			mem_list.append(x + [x[-1]] * (pad_mem - len(x)))
		else:
			mem_list.append(x[:pad_mem])
	MEM_arr = np.array(mem_list)
	#MEM_arr = np.array([x + [x[-1]] * (pad_mem - len(x)) for x in MEM])

	x_size = np.linspace(0, pad_size - 1, num=pad_size)
	x_mem = np.linspace(0, pad_mem - 1, num=pad_mem)

	return x_size, x_mem, SIZE_arr, MEM_arr


ALPHA = 0.2

COLOR1 = "#228B22"
COLOR2 = "#C3438F"
COLOR4 = "#191970"
COLOR3 = "#0060AD"
COLOR5 = "#A018CE"
COLOR6 = "#5E718B"
COLOR7 = "#049494"
COLOR8 = "#CF7041"
COLOR_NF = "#FFA500"

# ####################################################################################
# # PLOTTING EMPIRICAL GAME SIZE OVER TIME
# ####################################################################################

# x_size1, x_mem1, size1_arr, mem1_arr = plot_empirical_game_size_over_time(1)
# avg_size1 = np.mean(size1_arr, axis=0)
# std_size1 = np.std(size1_arr, axis=0)
# avg_mem1 = np.mean(mem1_arr, axis=0)
# std_mem1 = np.std(mem1_arr, axis=0)

# CONF_SIZE1 = 1.96 / math.sqrt(size1_arr.shape[0])
# CONF_MEM1 = 1.96 / math.sqrt(mem1_arr.shape[0])

# x_size2, x_mem2, size2_arr, mem2_arr = plot_empirical_game_size_over_time(2)
# avg_size2 = np.mean(size2_arr, axis=0)
# std_size2 = np.std(size2_arr, axis=0)
# avg_mem2 = np.mean(mem2_arr, axis=0)
# std_mem2 = np.std(mem2_arr, axis=0)

# CONF_SIZE2 = 1.96 / math.sqrt(size2_arr.shape[0])
# CONF_MEM2 = 1.96 / math.sqrt(mem2_arr.shape[0])

# x_size4, x_mem4, size4_arr, mem4_arr = plot_empirical_game_size_over_time(4)
# avg_size4 = np.mean(size4_arr, axis=0)
# std_size4 = np.std(size4_arr, axis=0)
# avg_mem4 = np.mean(mem4_arr, axis=0)
# std_mem4 = np.std(mem4_arr, axis=0)

# CONF_SIZE4 = 1.96 / math.sqrt(size4_arr.shape[0])
# CONF_MEM4 = 1.96 / math.sqrt(mem4_arr.shape[0])

# # x_size8, x_mem8, size8_arr, mem8_arr = plot_empirical_game_size_over_time(8)
# # avg_size8 = np.mean(size8_arr, axis=0)
# # std_size8 = np.std(size8_arr, axis=0)
# # avg_mem8 = np.mean(mem8_arr, axis=0)
# # std_mem8 = np.std(mem8_arr, axis=0)

# # CONF_SIZE8 = 1.96 / math.sqrt(size8_arr.shape[0])
# # CONF_MEM8 = 1.96 / math.sqrt(mem8_arr.shape[0])

# plt.rcParams.update({
#     "text.usetex": True, 
#     "font.family": "serif",
#     "axes.labelsize": 10,
#     "legend.fontsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
# })

# plt.figure(1, (6, 4))
# plt.plot(x_size1, avg_size1, color=COLOR5, label="M = 1")
# plt.fill_between(x_size1, avg_size1 - std_size1 * CONF_SIZE1, avg_size1 + std_size1 * CONF_SIZE1,
# 	alpha=ALPHA, facecolor=COLOR5)
# plt.plot(x_size2, avg_size2, color=COLOR6, label="M = 2")
# plt.fill_between(x_size2, avg_size2 - std_size2 * CONF_SIZE2, avg_size2 + std_size2 * CONF_SIZE2,
# 	alpha=ALPHA, facecolor=COLOR6)
# plt.plot(x_size4, avg_size4, color=COLOR7, label="M = 4")
# plt.fill_between(x_size4, avg_size4 - std_size4 * CONF_SIZE4, avg_size4 + std_size4 * CONF_SIZE4,
# 	alpha=ALPHA, facecolor=COLOR7)
# # plt.plot(x_size8, avg_size8, color=COLOR8, label="M = 8")
# # plt.fill_between(x_size8, avg_size8 - std_size8 * CONF_SIZE8, avg_size8 + std_size8 * CONF_SIZE8,
# # 	alpha=ALPHA, facecolor=COLOR8)

# # plt.xticks(np.arange(0, max(len(x_size1), len(x_size2), len(x_size4), len(x_size8)), step=5), weight='bold')
# plt.xticks(np.arange(0, max(len(x_size1), len(x_size2), len(x_size4)), step=5), weight='bold')
# plt.yticks(weight='bold')
# plt.xlabel(r"\textbf{TE-PSRO Epochs}")
# plt.ylabel(r"\textbf{Number of Information Sets}")
# plt.legend(loc="upper left")
# plt.show()

# plt.figure(2, (6, 4))
# plt.plot(x_mem1, avg_mem1, color=COLOR5, label="M = 1")
# plt.fill_between(x_mem1, avg_mem1 - std_mem1 * CONF_MEM1, avg_mem1 + std_mem1 * CONF_MEM1,
# 	alpha=ALPHA, facecolor=COLOR5)
# plt.plot(x_mem2, avg_mem2, color=COLOR6, label="M = 2")
# plt.fill_between(x_mem2, avg_mem2 - std_mem2 * CONF_MEM2, avg_mem2 + std_mem2 * CONF_MEM2,
# 	alpha=ALPHA, facecolor=COLOR6)
# plt.plot(x_mem4, avg_mem4, color=COLOR7, label="M = 4")
# plt.fill_between(x_mem4, avg_mem4 - std_mem4 * CONF_MEM4, avg_mem4 + std_mem4 * CONF_MEM4,
# 	alpha=ALPHA, facecolor=COLOR7)
# # plt.plot(x_mem8, avg_mem8, color=COLOR8, label="M = 8")
# # plt.fill_between(x_mem8, avg_mem8 - std_mem8 * CONF_MEM8, avg_mem8 + std_mem8 * CONF_MEM8,
# # 	alpha=ALPHA, facecolor=COLOR8)

# # plt.xticks(np.arange(0, max(len(x_mem1), len(x_mem2), len(x_mem4), len(x_mem8)), step=5), weight='bold')
# plt.xticks(np.arange(0, max(len(x_mem1), len(x_mem2), len(x_mem4)), step=5), weight='bold')
# plt.xlabel(r"\textbf{TE-PSRO Epochs}")
# # plt.ylabel("Size of Empirical Game in Gigabytes (GB)")
# plt.ylabel(r"\textbf{Megabytes (MB)}")
# plt.yticks(weight='bold')
# plt.legend(loc="upper left")
# plt.show()
# print(nah)
####################################################################################
# PLOTTING TRUE GAME REGRET -- 4 plots by MSS / EVAL COMBO
####################################################################################

x1_pbe_mss_ne_eval, REGRET_PBE_MSS_NE_EVAL_arr1 = plot_true_regret_over_time("ne", "pbe", 1, 13)
avg_regret1_PBE_MSS_NE_EVAL = np.mean(REGRET_PBE_MSS_NE_EVAL_arr1, axis=0)
std_regret1_PBE_MSS_NE_EVAL = np.std(REGRET_PBE_MSS_NE_EVAL_arr1, axis=0)

x2_pbe_mss_ne_eval, REGRET_PBE_MSS_NE_EVAL_arr2 = plot_true_regret_over_time("ne", "pbe", 2, 13)
avg_regret2_PBE_MSS_NE_EVAL = np.mean(REGRET_PBE_MSS_NE_EVAL_arr2, axis=0)
std_regret2_PBE_MSS_NE_EVAL = np.std(REGRET_PBE_MSS_NE_EVAL_arr2, axis=0)

x4_pbe_mss_ne_eval, REGRET_PBE_MSS_NE_EVAL_arr4 = plot_true_regret_over_time("ne", "pbe", 4, 13)
avg_regret4_PBE_MSS_NE_EVAL = np.mean(REGRET_PBE_MSS_NE_EVAL_arr4, axis=0)
std_regret4_PBE_MSS_NE_EVAL = np.std(REGRET_PBE_MSS_NE_EVAL_arr4, axis=0)

x8_pbe_mss_ne_eval, REGRET_PBE_MSS_NE_EVAL_arr8 = plot_true_regret_over_time("ne", "pbe", 8, 12)
avg_regret8_PBE_MSS_NE_EVAL = np.mean(REGRET_PBE_MSS_NE_EVAL_arr8, axis=0)
std_regret8_PBE_MSS_NE_EVAL = np.std(REGRET_PBE_MSS_NE_EVAL_arr8, axis=0)

x16_pbe_mss_ne_eval, REGRET_PBE_MSS_NE_EVAL_arr16 = plot_true_regret_over_time("ne", "pbe", 16, 17)
avg_regret16_PBE_MSS_NE_EVAL = np.mean(REGRET_PBE_MSS_NE_EVAL_arr16, axis=0)
std_regret16_PBE_MSS_NE_EVAL = np.std(REGRET_PBE_MSS_NE_EVAL_arr16, axis=0)


CONF_PBE_NE1 = 1.95 / 10
CONF_PBE_NE2 = 1.95 / 10
CONF_PBE_NE4 = 1.95 / 10
CONF_PBE_NE8 = 1.95 / 10
CONF_PBE_NE16 = 1.95 / 10

#'''
x1_ne_mss_ne_eval, REGRET_NE_MSS_NE_EVAL_arr1 = plot_true_regret_over_time("ne", "ne", 1, 13)
avg_regret1_NE_MSS_NE_EVAL = np.mean(REGRET_NE_MSS_NE_EVAL_arr1, axis=0)
std_regret1_NE_MSS_NE_EVAL = np.std(REGRET_NE_MSS_NE_EVAL_arr1, axis=0)

x2_ne_mss_ne_eval, REGRET_NE_MSS_NE_EVAL_arr2 = plot_true_regret_over_time("ne", "ne", 2, 13)
avg_regret2_NE_MSS_NE_EVAL = np.mean(REGRET_NE_MSS_NE_EVAL_arr2, axis=0)
std_regret2_NE_MSS_NE_EVAL = np.std(REGRET_NE_MSS_NE_EVAL_arr2, axis=0)

x4_ne_mss_ne_eval, REGRET_NE_MSS_NE_EVAL_arr4 = plot_true_regret_over_time("ne", "ne", 4, 13)
avg_regret4_NE_MSS_NE_EVAL = np.mean(REGRET_NE_MSS_NE_EVAL_arr4, axis=0)
std_regret4_NE_MSS_NE_EVAL = np.std(REGRET_NE_MSS_NE_EVAL_arr4, axis=0)
#'''
x8_ne_mss_ne_eval, REGRET_NE_MSS_NE_EVAL_arr8 = plot_true_regret_over_time("ne", "ne", 8, 13)
avg_regret8_NE_MSS_NE_EVAL = np.mean(REGRET_NE_MSS_NE_EVAL_arr8, axis=0)
std_regret8_NE_MSS_NE_EVAL = np.std(REGRET_NE_MSS_NE_EVAL_arr8, axis=0)

x16_ne_mss_ne_eval, REGRET_NE_MSS_NE_EVAL_arr16 = plot_true_regret_over_time("ne", "ne", 16, 15)
avg_regret16_NE_MSS_NE_EVAL = np.mean(REGRET_NE_MSS_NE_EVAL_arr16, axis=0)
std_regret16_NE_MSS_NE_EVAL = np.std(REGRET_NE_MSS_NE_EVAL_arr16, axis=0)

#'''
CONF_NE_NE1 = 1.95 / 10
CONF_NE_NE2 = 1.95 / 10
CONF_NE_NE4 = 1.95 / 10
CONF_NE_NE8 = 1.95 / 10
CONF_NE_NE16 = 1.95 / 10

####################################################################################
# PLOTTING TRUE GAME REGRET -- 2 plots by MSS = (NE or PBE), EVAL = NE
####################################################################################

plt.rcParams.update({
    "text.usetex": True, 
    "font.family": "serif",
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


plt.figure(1, (6, 4))
plt.plot(x1_pbe_mss_ne_eval, avg_regret1_PBE_MSS_NE_EVAL, color=COLOR1, label="M = 1")
plt.fill_between(x1_pbe_mss_ne_eval, avg_regret1_PBE_MSS_NE_EVAL - std_regret1_PBE_MSS_NE_EVAL * CONF_PBE_NE1, avg_regret1_PBE_MSS_NE_EVAL + std_regret1_PBE_MSS_NE_EVAL * CONF_PBE_NE1,
	alpha=ALPHA, facecolor=COLOR1)
plt.plot(x2_pbe_mss_ne_eval, avg_regret2_PBE_MSS_NE_EVAL, color=COLOR5, label="M = 2")
plt.fill_between(x2_pbe_mss_ne_eval, avg_regret2_PBE_MSS_NE_EVAL - std_regret2_PBE_MSS_NE_EVAL * CONF_PBE_NE2, avg_regret2_PBE_MSS_NE_EVAL + std_regret2_PBE_MSS_NE_EVAL * CONF_PBE_NE2,
	alpha=ALPHA, facecolor=COLOR5)
plt.plot(x4_pbe_mss_ne_eval, avg_regret4_PBE_MSS_NE_EVAL, color=COLOR3, label="M = 4")
plt.fill_between(x4_pbe_mss_ne_eval, avg_regret4_PBE_MSS_NE_EVAL - std_regret4_PBE_MSS_NE_EVAL * CONF_PBE_NE4, avg_regret4_PBE_MSS_NE_EVAL + std_regret4_PBE_MSS_NE_EVAL * CONF_PBE_NE4,
	alpha=ALPHA, facecolor=COLOR3)
plt.plot(x8_pbe_mss_ne_eval, avg_regret8_PBE_MSS_NE_EVAL, color=COLOR4, label="M = 8")
plt.fill_between(x8_pbe_mss_ne_eval, avg_regret8_PBE_MSS_NE_EVAL - std_regret8_PBE_MSS_NE_EVAL * CONF_PBE_NE8, avg_regret8_PBE_MSS_NE_EVAL + std_regret8_PBE_MSS_NE_EVAL * CONF_PBE_NE8,
	alpha=ALPHA, facecolor=COLOR4)
plt.plot(x16_pbe_mss_ne_eval, avg_regret16_PBE_MSS_NE_EVAL, color=COLOR8, label="M = 16")
plt.fill_between(x16_pbe_mss_ne_eval, avg_regret16_PBE_MSS_NE_EVAL - std_regret16_PBE_MSS_NE_EVAL * CONF_PBE_NE16, avg_regret16_PBE_MSS_NE_EVAL + std_regret16_PBE_MSS_NE_EVAL * CONF_PBE_NE16,
	alpha=ALPHA, facecolor=COLOR8)

plt.yticks(weight='bold')
plt.xticks(np.arange(0, max(len(x1_pbe_mss_ne_eval), len(x2_pbe_mss_ne_eval), len(x4_pbe_mss_ne_eval)), step=5), weight='bold')
plt.xlabel(r"\textbf{TE-PSRO Epochs}")
plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
plt.legend(loc='upper right')
plt.show()

####################################################################################
# PLOTTING TRUE GAME REGRET -- 5 plots by M = {1, 2, 4, 8, 16}
####################################################################################

plt.figure(1, (6, 4))
plt.plot(x1_ne_mss_ne_eval, avg_regret1_NE_MSS_NE_EVAL, color=COLOR1, label="Eval = NE, MSS = NE")
plt.fill_between(x1_ne_mss_ne_eval, avg_regret1_NE_MSS_NE_EVAL - std_regret1_NE_MSS_NE_EVAL * CONF_NE_NE1, avg_regret1_NE_MSS_NE_EVAL + std_regret1_NE_MSS_NE_EVAL * CONF_NE_NE1,
	alpha=ALPHA, facecolor=COLOR1)
plt.plot(x1_pbe_mss_ne_eval, avg_regret1_PBE_MSS_NE_EVAL, color=COLOR2, label="Eval = NE, MSS = PBE")
plt.fill_between(x1_pbe_mss_ne_eval, avg_regret1_PBE_MSS_NE_EVAL - std_regret1_PBE_MSS_NE_EVAL * CONF_PBE_NE1, avg_regret1_PBE_MSS_NE_EVAL + std_regret1_PBE_MSS_NE_EVAL * CONF_PBE_NE1,
	alpha=ALPHA, facecolor=COLOR2)
# plt.plot(x_nf, avg_regret_NF, color=COLOR_NF, label="NF")
# plt.fill_between(x_nf, avg_regret_NF - std_regret_NF * CONF_NF, avg_regret_NF + std_regret_NF * CONF_NF,
# 	alpha=ALPHA, facecolor=COLOR_NF)

plt.yticks(weight='bold')
#plt.xticks(np.arange(0, max(len(x1_spe_mss_ne_eval), len(x1_spe_mss_spe_eval), len(x1_ne_mss_ne_eval), len(x1_ne_mss_spe_eval), len(x_nf)), step=5), weight='bold')
# plt.xticks(np.arange(0, max(len(x1_pbe_mss_ne_eval), len(x1_ne_mss_ne_eval)), step=5), weight='bold')
plt.xticks(np.arange(0, 11, step=5), weight='bold')
plt.xlabel(r"\textbf{TE-PSRO Epochs}")
plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
plt.legend(loc='upper right')
plt.show()

#print(xxx)

plt.figure(2, (6, 4))
plt.plot(x2_ne_mss_ne_eval, avg_regret2_NE_MSS_NE_EVAL, color=COLOR1, label="Eval = NE, MSS = NE")
plt.fill_between(x2_ne_mss_ne_eval, avg_regret2_NE_MSS_NE_EVAL - std_regret2_NE_MSS_NE_EVAL * CONF_NE_NE2, avg_regret2_NE_MSS_NE_EVAL + std_regret2_NE_MSS_NE_EVAL * CONF_NE_NE2,
	alpha=ALPHA, facecolor=COLOR1)
plt.plot(x2_pbe_mss_ne_eval, avg_regret2_PBE_MSS_NE_EVAL, color=COLOR2, label="Eval = NE, MSS = PBE")
plt.fill_between(x2_pbe_mss_ne_eval, avg_regret2_PBE_MSS_NE_EVAL - std_regret2_PBE_MSS_NE_EVAL * CONF_PBE_NE2, avg_regret2_PBE_MSS_NE_EVAL + std_regret2_PBE_MSS_NE_EVAL * CONF_PBE_NE2,
	alpha=ALPHA, facecolor=COLOR2)
# plt.plot(x_nf, avg_regret_NF, color=COLOR_NF, label="NF")
# plt.fill_between(x_nf, avg_regret_NF - std_regret_NF * CONF_NF, avg_regret_NF + std_regret_NF * CONF_NF,
# 	alpha=ALPHA, facecolor=COLOR_NF)
plt.yticks(weight='bold')
# plt.xticks(np.arange(0, max(len(x2_spe_mss_ne_eval), len(x2_spe_mss_spe_eval), len(x2_ne_mss_ne_eval), len(x2_ne_mss_spe_eval), len(x_nf)), step=5), weight='bold')
# plt.xticks(np.arange(0, max(len(x2_pbe_mss_ne_eval), len(x2_ne_mss_ne_eval)), step=5), weight='bold')
plt.xticks(np.arange(0, 11, step=5), weight='bold')
plt.xlabel(r"\textbf{TE-PSRO Epochs}")
plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
plt.legend(loc='upper right')
plt.show()

plt.figure(3, (6, 4))
plt.plot(x4_ne_mss_ne_eval, avg_regret4_NE_MSS_NE_EVAL, color=COLOR1, label="Eval = NE, MSS = NE")
plt.fill_between(x4_ne_mss_ne_eval, avg_regret4_NE_MSS_NE_EVAL - std_regret4_NE_MSS_NE_EVAL * CONF_NE_NE4, avg_regret4_NE_MSS_NE_EVAL + std_regret4_NE_MSS_NE_EVAL * CONF_NE_NE4,
	alpha=ALPHA, facecolor=COLOR1)
plt.plot(x4_pbe_mss_ne_eval, avg_regret4_PBE_MSS_NE_EVAL, color=COLOR2, label="Eval = NE, MSS = PBE")
plt.fill_between(x4_pbe_mss_ne_eval, avg_regret4_PBE_MSS_NE_EVAL - std_regret4_PBE_MSS_NE_EVAL * CONF_PBE_NE4, avg_regret4_PBE_MSS_NE_EVAL + std_regret4_PBE_MSS_NE_EVAL * CONF_PBE_NE4,
	alpha=ALPHA, facecolor=COLOR2)
# plt.plot(x_nf, avg_regret_NF, color=COLOR_NF, label="NF")
# plt.fill_between(x_nf, avg_regret_NF - std_regret_NF * CONF_NF, avg_regret_NF + std_regret_NF * CONF_NF,
# 	alpha=ALPHA, facecolor=COLOR_NF)
plt.yticks(weight='bold')
# plt.xticks(np.arange(0, max(len(x4_spe_mss_ne_eval), len(x4_spe_mss_spe_eval), len(x4_ne_mss_ne_eval), len(x4_ne_mss_spe_eval), len(x_nf)), step=5), weight='bold')
# plt.xticks(np.arange(0, max(len(x4_pbe_mss_ne_eval), len(x4_ne_mss_ne_eval)), step=5), weight='bold')
plt.xticks(np.arange(0, 11, step=5), weight='bold')
plt.xlabel(r"\textbf{TE-PSRO Epochs}")
plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
plt.legend(loc='upper right')
plt.show()

plt.figure(4, (6, 4))
plt.plot(x8_ne_mss_ne_eval, avg_regret8_NE_MSS_NE_EVAL, color=COLOR1, label="Eval = NE, MSS = NE")
plt.fill_between(x8_ne_mss_ne_eval, avg_regret8_NE_MSS_NE_EVAL - std_regret8_NE_MSS_NE_EVAL * CONF_NE_NE8, avg_regret8_NE_MSS_NE_EVAL + std_regret8_NE_MSS_NE_EVAL * CONF_NE_NE8,
	alpha=ALPHA, facecolor=COLOR1)
plt.plot(x8_pbe_mss_ne_eval, avg_regret8_PBE_MSS_NE_EVAL, color=COLOR2, label="Eval = NE, MSS = PBE")
plt.fill_between(x8_pbe_mss_ne_eval, avg_regret8_PBE_MSS_NE_EVAL - std_regret8_PBE_MSS_NE_EVAL * CONF_PBE_NE8, avg_regret8_PBE_MSS_NE_EVAL + std_regret8_PBE_MSS_NE_EVAL * CONF_PBE_NE8,
	alpha=ALPHA, facecolor=COLOR2)
# plt.plot(x8_ne_mss_spe_eval, avg_regret8_NE_MSS_SPE_EVAL, color=COLOR3, label="Eval = SPE, MSS = NE")
# plt.fill_between(x8_ne_mss_spe_eval, avg_regret8_NE_MSS_SPE_EVAL - std_regret8_NE_MSS_SPE_EVAL * CONF_NE_SPE8, avg_regret8_NE_MSS_SPE_EVAL + std_regret8_NE_MSS_SPE_EVAL * CONF_NE_SPE8,
# 	alpha=ALPHA, facecolor=COLOR3)
# plt.plot(x8_spe_mss_spe_eval, avg_regret8_SPE_MSS_SPE_EVAL, color=COLOR4, label="Eval = SPE, MSS = SPE")
# plt.fill_between(x8_spe_mss_spe_eval, avg_regret8_SPE_MSS_SPE_EVAL - std_regret8_SPE_MSS_SPE_EVAL * CONF_SPE_SPE8, avg_regret8_SPE_MSS_SPE_EVAL + std_regret8_SPE_MSS_SPE_EVAL * CONF_SPE_SPE8,
# 	alpha=ALPHA, facecolor=COLOR4)
#plt.plot(x_nf, avg_regret_NF, color=COLOR_NF, label="NF")
#plt.fill_between(x_nf, avg_regret_NF - std_regret_NF * CONF_NF, avg_regret_NF + std_regret_NF * CONF_NF,
	# alpha=ALPHA, facecolor=COLOR_NF)
plt.yticks(weight='bold')
# plt.xticks(np.arange(0, max(len(x8_spe_mss_ne_eval), len(x8_spe_mss_spe_eval), len(x8_ne_mss_ne_eval), len(x8_ne_mss_spe_eval), len(x_nf)), step=5), weight='bold')
plt.xticks(np.arange(0, max(len(x8_pbe_mss_ne_eval), len(x8_ne_mss_ne_eval)), step=5), weight='bold')
plt.xlabel(r"\textbf{TE-PSRO Epochs}")
plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
plt.legend(loc='upper right')
plt.show()

plt.figure(5, (6, 4))
plt.plot(x16_ne_mss_ne_eval, avg_regret16_NE_MSS_NE_EVAL, color=COLOR1, label="Eval = NE, MSS = NE")
plt.fill_between(x16_ne_mss_ne_eval, avg_regret16_NE_MSS_NE_EVAL - std_regret16_NE_MSS_NE_EVAL * CONF_NE_NE16, avg_regret16_NE_MSS_NE_EVAL + std_regret16_NE_MSS_NE_EVAL * CONF_NE_NE16,
	alpha=ALPHA, facecolor=COLOR1)
plt.plot(x16_pbe_mss_ne_eval, avg_regret16_PBE_MSS_NE_EVAL, color=COLOR2, label="Eval = NE, MSS = PBE")
plt.fill_between(x16_pbe_mss_ne_eval, avg_regret16_PBE_MSS_NE_EVAL - std_regret16_PBE_MSS_NE_EVAL * CONF_PBE_NE16, avg_regret16_PBE_MSS_NE_EVAL + std_regret16_PBE_MSS_NE_EVAL * CONF_PBE_NE16,
	alpha=ALPHA, facecolor=COLOR2)
# plt.plot(x16_ne_mss_spe_eval, avg_regret16_NE_MSS_SPE_EVAL, color=COLOR3, label="Eval = SPE, MSS = NE")
# plt.fill_between(x16_ne_mss_spe_eval, avg_regret16_NE_MSS_SPE_EVAL - std_regret16_NE_MSS_SPE_EVAL * CONF_NE_SPE16, avg_regret16_NE_MSS_SPE_EVAL + std_regret16_NE_MSS_SPE_EVAL * CONF_NE_SPE16,
# 	alpha=ALPHA, facecolor=COLOR3)
# plt.plot(x16_spe_mss_spe_eval, avg_regret16_SPE_MSS_SPE_EVAL, color=COLOR4, label="Eval = SPE, MSS = SPE")
# plt.fill_between(x16_spe_mss_spe_eval, avg_regret16_SPE_MSS_SPE_EVAL - std_regret16_SPE_MSS_SPE_EVAL * CONF_SPE_SPE16, avg_regret16_SPE_MSS_SPE_EVAL + std_regret16_SPE_MSS_SPE_EVAL * CONF_SPE_SPE16,
# 	alpha=ALPHA, facecolor=COLOR4)
# plt.plot(x_nf, avg_regret_NF, color=COLOR_NF, label="NF")
# plt.fill_between(x_nf, avg_regret_NF - std_regret_NF * CONF_NF, avg_regret_NF + std_regret_NF * CONF_NF,
# 	alpha=ALPHA, facecolor=COLOR_NF)
plt.yticks(weight='bold')
# plt.xticks(np.arange(0, max(len(x16_spe_mss_ne_eval), len(x16_spe_mss_spe_eval), len(x16_ne_mss_ne_eval), len(x16_ne_mss_spe_eval), len(x_nf)), step=5), weight='bold')
plt.xticks(np.arange(0, max(len(x16_pbe_mss_ne_eval), len(x16_ne_mss_ne_eval)), step=5), weight='bold')
plt.xlabel(r"\textbf{TE-PSRO Epochs}")
plt.ylabel(r"\textbf{Average Regret of Solution in True Game}")
plt.legend(loc='upper right')
plt.show()




