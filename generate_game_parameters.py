import string
from abstract_games import *

game_settings = []

for i in range(15):

	CARD_WEIGHTS = {}
	for c in CHANCE_EVENTS:
		CARD_WEIGHTS[c] = float(random.randint(1, 10))

	payoff_map = {}
	for c in CHANCE_EVENTS:
		for a1 in P1_ACTIONS:
			for a2 in P2_ACTIONS:
				u = random.uniform(PAY_MIN, PAY_MAX)
				payoff_map[(c, a1, a2)] = [u, PAY_MAX - u]

	for R in range(1, NUM_ROUNDS):
		included_stochastic_rounds = list(range(R))
		file_ID = "abstract_" + str(NUM_ROUNDS) + "_" + str(included_stochastic_rounds[-1]) + "_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
		print("file_ID", file_ID)
		game_settings.append([file_ID, NUM_ROUNDS, P1_ACTIONS, P2_ACTIONS, CHANCE_EVENTS, CARD_WEIGHTS, payoff_map, included_stochastic_rounds])


np.savez_compressed('game_parameters_' + str(NUM_ROUNDS) + '_rounds', game_settings)

