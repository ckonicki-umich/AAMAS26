from best_response import *

def construct_initial_policy(game_param_map, hp_set):
	'''
	'''
	signals = ["H", "L"]
	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	offer_space = generate_offer_space(pool)
	offer_space1 = offer_space[:]
	offer_space1.remove(("deal",))
	offer_space1.remove(("walk",))

	initial_policy = {}

	POLICY_SPACE1 = {}
	POLICY_SPACE2 = {}

	state_len = get_state_length(pool)
	action_space = list(it.product(offer_space, [True, False]))
	action_space1_start = list(it.product(offer_space1, [True, False]))

	default_start_model1 = tf.keras.Sequential()
	default_start_model1.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_start_model1.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))
	default_start_model1.add(tf.keras.layers.Dense(len(action_space1_start)))
	default_start_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))
	W1_start = default_start_model1.get_weights()

	default_model1 = tf.keras.Sequential()
	default_model1.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_model1.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))
	default_model1.add(tf.keras.layers.Dense(len(action_space)))
	default_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))

	default_model2 = tf.keras.Sequential()
	default_model2.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_model2.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))
	default_model2.add(tf.keras.layers.Dense(len(action_space)))
	default_model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))

	W2_default = default_model2.get_weights()

	reveal1L, reveal1H, reveal1, reveal2 = np.random.choice([True, False], 4)

	initial_policy[(1, ("H",))] = {("pi_0", reveal1H): 1.0}
	#initial_policy[(1, ("H",))] = {("pi_0", reveal1H): 0.8, ("pi_0", not reveal1H): 0.2}
	initial_policy[(1, ("L",))] = {("pi_0", reveal1L): 1.0}
	#initial_policy[(1, ("L",))] = {("pi_0", reveal1L): 1.0 / 3, ("pi_0", not reveal1L): 2.0 / 3}

	if reveal1H:
		initial_policy[(2, ("H", "H", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("H", "L", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}
	else:
		initial_policy[(2, ("H", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("L", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}

	if reveal1L:
		initial_policy[(2, ("L", "H", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("L", "L", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}
	else:
		initial_policy[(2, ("H", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("L", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}

	POLICY_SPACE1["pi_0"] = W1_start
	POLICY_SPACE2["pi_0"] = W2_default
	
	return initial_policy, ("pi_0", bool(reveal1)), ("pi_0", bool(reveal2)), POLICY_SPACE1, POLICY_SPACE2