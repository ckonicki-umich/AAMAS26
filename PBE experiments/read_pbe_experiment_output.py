import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams
from config import Config

config = Config.load_config("pbe_algorithm_config.yaml")

def gen_arrays(directory_of_result_files):
	'''
	@arg (str) directory_of_result_files: folder containing all saved .npz from experiment

	Organizes the experiment outputs into five arrays: 2 runtime arrays for
	CFR and PBE, 2 worst-case subgame regrets for CFR anfile_list, d PBE, and one array
	for empirical game size
	'''
	RUNTIME_CFR_500 = []
	RUNTIME_CFR_1000 = []
	RUNTIME_CFR_2000 = []
	RUNTIME_CFR_5000 = []

	RUNTIME_PBE_500 = []
	RUNTIME_PBE_1000 = []
	RUNTIME_PBE_2000 = []
	RUNTIME_PBE_5000 = []

	INFOSET_PBE_REGRET_500 = []
	INFOSET_PBE_REGRET_1000 = []
	INFOSET_PBE_REGRET_2000 = []
	INFOSET_PBE_REGRET_5000 = []

	GAME_SIZE = []

	for filename in os.listdir(directory_of_result_files):
		full_path = os.path.join(directory_of_result_files, filename)
		assert os.path.isfile(full_path)

		a_f = np.load(full_path, allow_pickle=True)
		for output_i in a_f['arr_0']:
			for i in range(len(output_i)):
				t = output_i[i][0]
				regret = float(output_i[i][1])

				if t == '500':
					INFOSET_PBE_REGRET_500.append(regret)
				elif t == '1000':
					INFOSET_PBE_REGRET_1000.append(regret)
				elif t == '2000':
					INFOSET_PBE_REGRET_2000.append(regret)
				else:
					INFOSET_PBE_REGRET_5000.append(regret)

		for i in range(len(a_f['arr_4'])):

			output_i = a_f['arr_4'][i]
			game_size = a_f['arr_1'][i, 0]
			GAME_SIZE.append(game_size)
			for i in range(len(output_i[1])):
				t = output_i[1][i]

				if t[0] == "500":
					RUNTIME_PBE_500.append(t[1] / 60.0)
					RUNTIME_CFR_500.append(t[2] / 60.0)
				elif t[0] == "1000":
					RUNTIME_PBE_1000.append(t[1] / 60.0)
					RUNTIME_CFR_1000.append(t[2] / 60.0)
				elif t[0] == "2000":
					RUNTIME_PBE_2000.append(t[1] / 60.0)
					RUNTIME_CFR_2000.append(t[2] / 60.0)
				else:
					RUNTIME_PBE_5000.append(t[1] / 60.0)
					RUNTIME_CFR_5000.append(t[2] / 60.0)

		RUNTIME_CFR = np.array([RUNTIME_CFR_500, RUNTIME_CFR_1000, RUNTIME_CFR_2000, RUNTIME_CFR_5000])
		RUNTIME_PBE = np.array([RUNTIME_PBE_500, RUNTIME_PBE_1000, RUNTIME_PBE_2000, RUNTIME_PBE_5000])
		INFOSET_PBE_REGRET = np.array([INFOSET_PBE_REGRET_500, INFOSET_PBE_REGRET_1000, INFOSET_PBE_REGRET_2000, INFOSET_PBE_REGRET_5000])

	return RUNTIME_CFR, RUNTIME_PBE, INFOSET_PBE_REGRET, GAME_SIZE

def perform_regression(runtime_list):
	'''
	'''
	r_500 = runtime_list[0]
	r_1000 = runtime_list[1]
	r_2000 = runtime_list[2]
	r_5000 = runtime_list[3]

	def helper(x, y):
		x_arr = np.array(x).reshape((-1, 1))
		y_arr = np.array(y)
		model = LinearRegression(fit_intercept=False).fit(x_arr, y_arr)
		
		return model.score(x_arr, y_arr), model.coef_[0]

	# coef should be close to 2
	score_1000, coef_1000 = helper(r_500, r_1000)
	print("r2_1000 ", score_1000)
	print("coef_1000 ", coef_1000)

	# coef should be close to 4
	score_2000, coef_2000 = helper(r_500, r_2000)
	print("r2_2000 ", score_2000)
	print("coef_2000 ", coef_2000)

	# coef should be close to 10
	score_5000, coef_5000 = helper(r_500, r_5000)
	print("r2_5000 ", score_5000)
	print("coef_5000 ", coef_5000)

	return None

prefix4 = "res_pbe_exp_4rounds/"
prefix5 = "res_pbe_exp_5rounds/"

T_list = config.T_list

rt_cfr4, rt_pbe4, pbe_regrets4, game_size4 = gen_arrays(prefix4)
print(game_size4[0])


print("NE regression")
perform_regression(rt_cfr4)
print("\n")
print("PBE regression")
perform_regression(rt_pbe4)
print("\n")

print("PBE regrets 4-rounds")
for i in range(len(pbe_regrets4)):
	print("T = ", T_list[i])
	print(np.mean(pbe_regrets4[i]))

rt_cfr5, rt_pbe5, pbe_regrets5, game_size5 = gen_arrays(prefix5)

print("NE regression 5-rounds")
perform_regression(rt_cfr5)
print("\n")
print("PBE regression 5-rounds")
perform_regression(rt_pbe5)
print("\n")

print("PBE regrets 5-rounds")
for i in range(len(pbe_regrets5)):
	print("T = ", T_list[i])
	print(np.mean(pbe_regrets5[i]))

plt.rcParams.update({
    "text.usetex": True, 
    "font.family": "serif",
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PBE_COLOR = "#CD5B45"
NE_COLOR = "#3A8953"

runtime_R4_CFR_T500 = rt_cfr4[0]
runtime_R4_CFR_T1000 = rt_cfr4[1]
runtime_R4_CFR_T2000 = rt_cfr4[2]
runtime_R4_CFR_T5000 = rt_cfr4[3]

runtime_R4_PBE_T500 = rt_pbe4[0]
runtime_R4_PBE_T1000 = rt_pbe4[1]
runtime_R4_PBE_T2000 = rt_pbe4[2]
runtime_R4_PBE_T5000 = rt_pbe4[3]

runtime_R5_CFR_T500 = rt_cfr5[0]
runtime_R5_CFR_T1000 = rt_cfr5[1]
runtime_R5_CFR_T2000 = rt_cfr5[2]
runtime_R5_CFR_T5000 = rt_cfr5[3]

runtime_R5_PBE_T500 = rt_pbe5[0]
runtime_R5_PBE_T1000 = rt_pbe5[1]
runtime_R5_PBE_T2000 = rt_pbe5[2]
runtime_R5_PBE_T5000 = rt_pbe5[3]

plt.figure(1, (6, 4))
plt.scatter(game_size4, runtime_R4_PBE_T500, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size4, runtime_R4_CFR_T500, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")

plt.figure(2, (6, 4))
plt.scatter(game_size4, runtime_R4_PBE_T1000, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size4, runtime_R4_CFR_T1000, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")


plt.figure(3, (6, 4))
plt.scatter(game_size4, runtime_R4_PBE_T2000, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size4, runtime_R4_CFR_T2000, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")


plt.figure(4, (6, 4))
plt.scatter(game_size4, runtime_R4_PBE_T5000, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size4, runtime_R4_CFR_T5000, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")

plt.figure(1, (6, 4))
plt.scatter(game_size5, runtime_R5_PBE_T500, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size5, runtime_R5_CFR_T500, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")

plt.figure(2, (6, 4))
plt.scatter(game_size5, runtime_R5_PBE_T1000, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size5, runtime_R5_CFR_T1000, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")


plt.figure(3, (6, 4))
plt.scatter(game_size5, runtime_R5_PBE_T2000, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size5, runtime_R5_CFR_T2000, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")


plt.figure(4, (6, 4))
plt.scatter(game_size5, runtime_R5_PBE_T5000, s=15, c=PBE_COLOR, alpha=1.0, label="PBE")
plt.scatter(game_size5, runtime_R5_CFR_T5000, s=15, c=NE_COLOR, alpha=1.0, label="NE")
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel(r"\textbf{Number of Information Sets in Game}", fontsize=12)
plt.ylabel(r"\textbf{Runtime in Minutes}", fontsize=12)
plt.legend(fontsize="9", loc="upper left")

plt.show()



