from best_response import *

def construct_initial_policy(game_param_map, hp_set1, hp_set2):
	'''
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]

	initial_policy = {}
	POLICY_SPACE1 = {}
	POLICY_SPACE2 = {}

	state_len = get_state_length(num_rounds)
	action_space1 = p1_actions[:]
	default_model1 = tf.keras.Sequential()

	default_model1.add(tf.keras.layers.Dense(hp_set1["model_width"], input_shape=(state_len,), activation="relu"))
	default_model1.add(tf.keras.layers.Dense(hp_set1["model_width"], activation="relu"))

	default_model1.add(tf.keras.layers.Dense(len(action_space1)))
	default_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set1["learning_rate"]))

	W1_default = default_model1.get_weights()

	action_space2 = p2_actions[:]
	default_model2 = tf.keras.Sequential()

	default_model2.add(tf.keras.layers.Dense(hp_set2["model_width"], input_shape=(state_len,), activation="relu"))
	default_model2.add(tf.keras.layers.Dense(hp_set2["model_width"], activation="relu"))

	default_model2.add(tf.keras.layers.Dense(len(action_space2)))
	default_model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set2["learning_rate"]))

	W2_default = default_model2.get_weights()

	POLICY_SPACE1["pi_0"] = W1_default
	POLICY_SPACE2["pi_0"] = W2_default

	for card in chance_events:
		initial_policy[(1, (card,))] = {"pi_0": 1.0}
		initial_policy[(2, (card, "pi_0",))] = {"pi_0": 1.0}

	cur_policy = initial_policy.copy()
	infoset_histories = []
	for infoset_id in cur_policy:
		if infoset_id[0] == 2 and infoset_id[1] not in infoset_histories:
			infoset_histories.append(infoset_id[1])

	for r in range(num_rounds - 2):
		next_policy = {}
		for infoset_history in infoset_histories:
			empir_history = infoset_history + ("pi_0",)

			for card in chance_events:
				if card not in empir_history:
					next_empir_history = empir_history + (card,)
					next_policy[(1, next_empir_history)] = {"pi_0": 1.0}
					initial_policy[(1, next_empir_history)] = {"pi_0": 1.0}

					next_empir_history = next_empir_history + ("pi_0",)
					next_policy[(2, next_empir_history)] = {"pi_0": 1.0}
					initial_policy[(2, next_empir_history)] = {"pi_0": 1.0}

		cur_policy = next_policy.copy()
		infoset_histories = []
		for infoset_id in cur_policy:
			if infoset_id[0] == 2 and infoset_id[1] not in infoset_histories:
				infoset_histories.append(infoset_id[1])

	extra_copy = initial_policy.copy()
	initial_policy = {}
	for infoset_id in extra_copy:
		x = infoset_id[1]
		chance_ind = [j for j in range(len(x)) if x[j] in chance_events][-1]
		new_key = None
		if infoset_id[0] == 1:
			new_key = (1, x[:chance_ind])
		else:
			new_key = (2, x[:chance_ind] + x[chance_ind+1:])

		initial_policy[new_key] = extra_copy[infoset_id]

	return initial_policy, "pi_0", "pi_0", POLICY_SPACE1, POLICY_SPACE2
