import itertools as it
from Node import *
from Infoset import *
import networkx as nx

ALPHA = 0.1

class ExtensiveForm:
	'''
	Implementation of Extensive Form object
	'''
	def __init__(self, infosets, root_node, terminal_nodes, chance_map, num_rounds):
		'''
		@arg (list of lists of Infosets) infosets: Collection I of information sets for each player, represented
			as a list of each player's Infoset objects, 1 through N
		@arg (list of Nodes) terminal_nodes: Collection T of terminal_nodes, represented as a list of Node
			objects
		@arg (list of dicts) strategy_space: List of initial pure strategies for each player, at each infoset;
			each dictionary (one per player minus Nature) in the list maps a player's infoset object to an action
			in the infoset's action space
		@arg (dict) chance_map: Map of each chance Node object h to the set of actions available to Nature
			at node h X(h)
		'''
		self.infosets = infosets
		self.terminal_nodes = terminal_nodes
		self.root = root_node
		self.chance_map = chance_map
		self.num_players = len(self.infosets)
		self.num_rounds = num_rounds

	def get_infoset_given_node(self, node):
		'''
		@arg (Node) node: given node in the game tree

		Helper method to find the infoset that contains the input node
		'''
		matching_infosets = [x for x in self.infosets[node.player_id - 1] if x.infoset_id == node.infoset_id]
		if len(matching_infosets) == 0:
			return None
		return matching_infosets[0]

	def compute_pay(self, strategy_profile):
		'''
		@arg (map: Infoset --> (map: str --> float)) strategy_profile: each key
			in the outer map is a player infoset. Each player infoset's strategy is represented
			as a second map giving a distribution over that infoset's action space

		Compute the expected payoff for all players, given a joint strategy profile
		'''
		return self.root.compute_pay(strategy_profile, self.chance_map, 1.0)

	def compute_reach_prob(self, strategy_profile, node):
		'''
		@arg (list of list of infoset, strategy map pairs) strategy_profile: each elt
			in the list is a player's strategy. Each player's strategy is represented
			as a list of tuples: (infoset, map giving a distribution over that infoset's
			action space)
		@arg (Node) node: Node object

		Compute the reach probability for a given node in the tree
		'''
		reach_prob = 1.0
		current = self.root

		for h in node.history:

			if current.player_id == 0:
				event_map = self.get_prob_dist_given_chance_map(current)
				if event_map.get(h) == 0.0:
					return 0.0
				reach_prob *= event_map.get(h)
			else:
				j = current.player_id - 1
				matching = [infoset_id for infoset_id in strategy_profile.keys() if infoset_id == current.infoset_id]
				if matching == []:
					matching_tups = [tup for tup in strategy_profile.keys() if current.infoset_id in tup]
					if len(matching_tups) == 0:
						return 0.0

					matching_tup = matching_tups[0]
					k = matching_tup.index(current.infoset_id)
					current_strategy = strategy_profile.get(matching_tup)
					action = [x for x in current_strategy.keys() if x[k] == h]

					if action == []:
						return 0.0

					action_ = action[0]
					reach_prob *= current_strategy.get(action_, 0.0)
				else:
					current_infoset_id = matching[0]
					current_strategy = strategy_profile.get(current_infoset_id)
					if current_strategy.get(h, 0.0) == 0.0:
						return 0.0
					reach_prob *= current_strategy.get(h, 0.0)

					# if we're using belief states
					if current.use_beliefs:
						reach_prob *= current.belief

			next_node = current.get_child_given_action(h)
			current = next_node

		return reach_prob

	def get_next_infoset_id(self, path, i):
		'''
		'''
		if len(path[i+1]) == 1:
			return 0

		num_player_actions = len([x for x in path[:(i+1)] if len(x) > 1])
		next_player_id = num_player_actions % 2 + 1
		next_infoset_id = None
		if next_player_id == 1:
			empir_history = tuple(path)[:i]
			next_infoset_id = (1, empir_history)
		else:
			empir_history = tuple(path)[:(i - 1)] + (path[i],)
			next_infoset_id = (2, empir_history)

		return next_infoset_id

	def update_game_with_simulation_output(self, observations, payoffs):
		'''
		Helper method intended for the EMPIRICAL game. Update the empirical game with new info
		from the simulator resulting from simulating a given strategy profile. This is how we 
		add brand new untraveled paths to the game tree

		This is how we update the empirical leaf utilities and Nature's empirical probability distributions.
		'''
		for tup_path in observations:
			path = list(tup_path)
			cur_node = self.root

			for i in range(len(path)):
				a = path[i]

				if i == (len(path) - 1):
					comp_history = path[:]
					term = Node(None, (None, None), comp_history, None, 2)
					term.make_terminal(np.zeros(2))
					
					def gen_matching_nodes(children, comp_history):
						for x in children:
							if x.history == comp_history:
								yield x

					def gen_matching_terminal_nodes(terminal_nodes, comp_history):
						for x in terminal_nodes:
							if x.history[:(i+1)] == comp_history:
								yield x

					matching = list(gen_matching_nodes(cur_node.children, comp_history))
					matching_terminal_nodes = list(gen_matching_terminal_nodes(self.terminal_nodes, comp_history))
					
					# make sure there's no identical terminal nodes with matching history AND that 
					# this possible new terminal node's history isn't already covered by a decision node
					if len(matching) == 0 and len(matching_terminal_nodes) == 0:
						cur_node.add_children([term])
						self.terminal_nodes.append(term)

					if len(matching_terminal_nodes) > 0:
						for x in matching_terminal_nodes:
							payoffs[tuple(x.history)] = payoffs.get(tuple(x.history), []) + payoffs.get(tuple(comp_history), [])

					for j in range(i):
						h = path[:j+1]
						old_payoffs = payoffs.get(tuple(h), [])
						if old_payoffs != []:
							new_payoffs = payoffs.get(tuple(comp_history), []) + old_payoffs
							payoffs[tuple(comp_history)] = new_payoffs[:]
							del payoffs[tuple(h)]

					cur_infoset = self.get_infoset_given_node(cur_node)
					if a not in cur_infoset.action_space:
						cur_infoset.action_space += [a]
						for n in cur_infoset.node_list:
							n.action_space = cur_infoset.action_space[:]
					
					if a not in cur_node.action_space:
						cur_node.action_space += [a]
				
				elif i % 3 == 0 and len(a) == 1:
					def get_matching_children(children, a):
						for x in children:
							if x.history[-1] == a:
								yield x

					children_matching = list(get_matching_children(cur_node.children, a))
					if len(children_matching) == 0:
						# New path is being formulated
						cur_node.action_space += [a]
						if cur_node in self.chance_map.keys():
							cur_dist = self.chance_map[cur_node].copy()
							cur_dist[a] = cur_dist.get(a, 0.0) + observations.get(tup_path)
							self.chance_map[cur_node] = cur_dist
						else:
							self.chance_map[cur_node] = {a : observations.get(tup_path)}

						next_player_id = 1
						next_infoset_id = (1, tuple(path)[:i])
						next_node = Node(next_player_id, next_infoset_id, cur_node.history + [a], [], 2)
						cur_node.add_children([next_node])

						def gen_matching_infosets(infosets, next_infoset_id):
							for x in infosets:
								if x.infoset_id == next_infoset_id:
									yield x

						matching_infosets = list(gen_matching_infosets(self.infosets[0], next_infoset_id))

						if matching_infosets == []:
							next_infoset = Infoset(next_infoset_id, [next_node], [], 2)
							self.infosets[next_player_id - 1].append(next_infoset)
						else:
							for matching_infoset in matching_infosets:
								next_node.action_space = matching_infoset.action_space[:]
								matching_infoset.node_list += [next_node]

					else:
						path_to_match = path[:i]
						if cur_node in self.chance_map.keys():
							cur_dist = self.chance_map[cur_node].copy()
							cur_dist[a] = cur_dist.get(a, 0.0) + observations.get(tup_path)
							self.chance_map[cur_node] = cur_dist
						else:
							self.chance_map[cur_node] = {a : observations.get(tup_path)}

						def gen_next_nodes(children, a, path_to_match):
							for x in children:
								if a == x.history[-1] and list(x.infoset_id[1]) == path_to_match:
									yield x

						next_nodes = list(gen_next_nodes(cur_node.children, a, path_to_match))
						next_node = next_nodes[0]

					cur_node = next_node
							
				else:

					def get_matching_children(path, i, cur_node):
						for x in cur_node.children:
							if x.history == path[:(i+1)]:
								yield x

					def get_nonterminal_matching_children(children_matching):
						for x in children_matching:
							if x not in self.terminal_nodes:
								yield x

					children_matching = list(get_matching_children(path, i, cur_node))
					non_terminal_children_matching = list(get_nonterminal_matching_children(children_matching))
					cur_player_id = cur_node.infoset_id[0]
					
					if len(children_matching) == 0:
						# New path is being formulated
						assert cur_player_id != 0
						cur_infoset = self.get_infoset_given_node(cur_node)

						if a not in cur_infoset.action_space:
							cur_infoset.action_space.append(a)
							for n in cur_infoset.node_list:
								n.action_space = cur_infoset.action_space[:]

						if a not in cur_node.action_space:
							cur_node.action_space.append(a)

						next_infoset_id = self.get_next_infoset_id(path, i)

						def get_infoset_given_id(next_infoset_id):
							j = next_infoset_id[0]
							for x in self.infosets[j - 1]:
								if x.infoset_id == next_infoset_id:
									yield x

						if next_infoset_id == 0:
							next_infoset_id = (0, len(self.chance_map))
							next_node = Node(0, next_infoset_id, cur_node.history + [a], [], 2)
						else:
							next_player_id = next_infoset_id[0]
							next_node = Node(next_player_id, next_infoset_id, cur_node.history + [a], [], 2)
							matching_infosets = list(get_infoset_given_id(next_infoset_id))
							
							if matching_infosets == []:
								next_infoset = Infoset(next_infoset_id, [next_node], [], 2)
								self.infosets[next_player_id - 1].append(next_infoset)
							else:
								for matching_infoset in matching_infosets:
									next_node.action_space = matching_infoset.action_space[:]
									matching_infoset.node_list += [next_node]
								assert len(matching_infosets) == 1

						cur_node.add_children([next_node])

					elif len(non_terminal_children_matching) == 0:
						def gen_matching_children(children, a):
							for x in children:
								if x.history[-1] == a:
									yield x

						next_nodes = list(gen_matching_children(cur_node.children, a))
						assert len(next_nodes) == 1
						next_node = next_nodes[0]
						assert next_node in children_matching
						current_utility = next_node.utility

						next_infoset_id = self.get_next_infoset_id(path, i)
						if next_infoset_id == 0:
							next_infoset_id = (0, len(self.chance_map))

						next_player_id = next_infoset_id[0]
						next_node.is_terminal = False
						next_node.player_id = next_player_id
						next_node.infoset_id = next_infoset_id
						next_node.action_space = []
						self.terminal_nodes.remove(next_node)

						if next_player_id != 0:
							def get_infoset_given_id(next_infoset_id):
								j = next_infoset_id[0]
								for x in self.infosets[j - 1]:
									if x.infoset_id == next_infoset_id:
										yield x

							matching_infosets = list(get_infoset_given_id(next_infoset_id))
							if matching_infosets == []:
								next_infoset = Infoset(next_infoset_id, [next_node], [], 2)
								self.infosets[next_player_id - 1].append(next_infoset)
							else:
								assert len(matching_infosets) == 1
								for matching_infoset in matching_infosets:
									next_node.action_space = matching_infoset.action_space[:]
									matching_infoset.node_list += [next_node]

						if np.any(current_utility):
							payoffs[tuple(next_node.history)] = payoffs.get(tuple(next_node.history), []) + [current_utility]

					else:
						def gen_matching_children(children, a):
							for x in children:
								if x.history[-1] == a:
									yield x
						
						next_nodes = list(gen_matching_children(cur_node.children, a))
						assert len(next_nodes) == 1
						next_node = next_nodes[0]

						if cur_player_id == 0:
							cur_dist = self.chance_map[cur_node].copy()
							cur_dist[a] = cur_dist.get(a, 0.0) + 1
							self.chance_map[cur_node] = cur_dist

					cur_node = next_node

		for t in self.terminal_nodes:
			payoffs_t = [payoffs[x] for x in payoffs.keys() if x == tuple(t.history)]
			if payoffs_t != []:
				payoffs_t = [payoffs[x] for x in payoffs.keys() if x == tuple(t.history)][0]
				t.utility = np.sum(payoffs_t, axis=0) / len(payoffs_t)

	def get_prob_dist_given_chance_map(self, node):
		'''
		'''
		card_weights = self.chance_map.get(node).copy()
		prob_dist = {}
		denom = sum(card_weights.values())
		for e in card_weights.keys():
			prob_dist[e] = card_weights.get(e) / denom

		return prob_dist

	def cfr(self, T):
		'''
		General implementation of counterfactual regret minimization (CFR).
		This method is to be called by the empirical game ExtensiveForm object
		Returns a new metastrategy that should ultimately be an approx. NE

		Later: could modify to include beliefs?
		'''
		print("called regular CFR")

		# First, initialize key variables
		for j in range(2):
			for infoset in self.infosets[j]:
				num_actions = len(infoset.action_space)
				infoset.set_strategy(np.repeat(1.0 / num_actions, num_actions))
				infoset.regret_sum = np.zeros(num_actions)
				infoset.strategy_sum = np.zeros(num_actions)
				infoset.reach_prob_sum = 0
				infoset.reach_prob = 0
				infoset.action_utils = np.zeros((num_actions, 2))

		expected_val_cur_strategy = np.zeros(2)
		for t in range(T):
			if t % 100 == 0:
				print("t ", t)

			expected_val_cur_strategy += self.recursive_cfr_helper(self.root, 1.0, 1.0, 1.0)
			for j in range(2):
				for infoset in self.infosets[j]:
					infoset.update_strategy()

		# computing average strat --> Nash equil
		nash_strat = {}
		for j in range(2):
			for infoset in self.infosets[j]:
				nash_I = infoset.compute_average_strategy()
				num_actions = len(infoset.action_space)
				dist = {}
				for i in range(num_actions):
					a = infoset.action_space[i]
					dist[a] = nash_I[i]

				nash_strat[infoset.infoset_id] = dist

		return nash_strat

	def recursive_cfr_helper(self, current_node, player1_prob, player2_prob, chance_prob, partial_solution={}):
		'''
		@arg (Node) current_node: node within a current information set we are currently visiting as we
			play the game
		@arg (Infoset) current_infoset: the current information set we're visiting

		@arg (float) player1_prob: the reach probability contributed by player 1
		@arg (float) player2_prob: the reach probability contributed by player 2
		@arg (float) chance_prob: the reach probability contributed by Nature

		Recursive helper function that updates action utilities, computes the counterfactual utilities
			of the current strategy, and updates cumulative regret at that information set in turn
		'''
		if current_node.player_id == 0:
			expected_pay = np.zeros(2)
			prob_dist = self.get_prob_dist_given_chance_map(current_node)
			for outcome in prob_dist.keys():
				next_node = current_node.get_child_given_action(outcome)
				next_pay = self.recursive_cfr_helper(next_node, player1_prob, player2_prob, chance_prob * prob_dist.get(outcome), partial_solution)
				expected_pay += next_pay * prob_dist.get(outcome) / sum(prob_dist.values())

			return expected_pay

		elif current_node.is_terminal:
			return current_node.utility

		elif current_node.infoset_id in partial_solution:
			return current_node.compute_pay(partial_solution, self.chance_map, 1.0)

		# Now we want to compute the counterfactual utility
		current_infoset = self.get_infoset_given_node(current_node)
		num_avail_actions = len(current_infoset.action_space)
		current_strategy = current_infoset.strategy

		# now try this to fix reach_prob being > 1.0 for player 2 at times
		if current_node.player_id == 1:
			current_infoset.reach_prob += player1_prob
		else:
			current_infoset.reach_prob += player2_prob

		infoset_action_utils = np.zeros((num_avail_actions, 2))
		for i in range(num_avail_actions):
			a = current_infoset.action_space[i]
			next_node = current_node.get_child_given_action(a)

			if next_node is not None:
				if current_node.player_id == 1:
					# updated player 1 
					infoset_action_utils[i] = self.recursive_cfr_helper(next_node, player1_prob * current_strategy[i], player2_prob, chance_prob, partial_solution)
				else:
					# updated player 2
					infoset_action_utils[i] = self.recursive_cfr_helper(next_node, player1_prob, player2_prob * current_strategy[i], chance_prob, partial_solution)

			else:
				if current_node.player_id == 1:
					infoset_action_utils[i] = np.array([0.0, 0.0])
				else:
					infoset_action_utils[i] = np.array([0.0, 0.0])

		# Now compute the total utility of the information set
		infoset_cfu = np.matmul(current_strategy, infoset_action_utils)

		# Compute the regrets of not playing each action at the infoset
		regrets = infoset_action_utils - infoset_cfu

		if current_node.player_id == 1:
			current_infoset.regret_sum += regrets[:, 0] * player2_prob * chance_prob
		else:
			current_infoset.regret_sum += regrets[:, 1] * player1_prob * chance_prob

		return infoset_cfu

	def recursive_pbe_cfr_helper(self, current_node, player_reach_probs, beliefs, expected_utilities):
		'''
		'''
		if current_node.is_terminal:
			return current_node.utility

		elif current_node.player_id == 0:
			expected_pay = np.zeros(self.num_players)
			prob_dist = self.get_prob_dist_given_chance_map(current_node)

			for outcome in prob_dist.keys():
				next_node = current_node.get_child_given_action(outcome)
				new_player_reach_probs = player_reach_probs.copy()
				new_player_reach_probs[0] *= prob_dist.get(outcome)
				expected_utilities[next_node] = self.recursive_pbe_cfr_helper(next_node, new_player_reach_probs, beliefs, expected_utilities)
				expected_pay += prob_dist.get(outcome) * expected_utilities[next_node]

			return expected_pay

		# Now we want to compute the immediate believed counterfactual utility
		current_infoset = self.get_infoset_given_node(current_node)
		num_avail_actions = len(current_infoset.action_space)
		current_strategy = current_infoset.strategy
		for child in current_node.children:
			expected_utilities[child] = np.zeros(self.num_players)

		expected_utilities[current_node] = np.zeros(self.num_players)
		current_infoset.reach_prob += player_reach_probs[current_node.player_id]
		infoset_belief = beliefs.get(current_node.infoset_id)
		
		# U^B(strat, beliefs | I, a) for all a
		believed_infoset_action_utils = np.zeros((num_avail_actions, self.num_players))
		expected_pay_h = np.zeros(self.num_players)

		# obtain children's instant cf val of I
		for i in range(num_avail_actions):
			a = current_infoset.action_space[i]
			assert a in current_node.action_space
			next_node = current_node.get_child_given_action(a)

			if next_node is not None:
				new_player_reach_probs = player_reach_probs.copy()
				new_player_reach_probs[current_node.player_id] = player_reach_probs[current_node.player_id] * current_strategy[i]
				expected_utilities[next_node] = self.recursive_pbe_cfr_helper(next_node, new_player_reach_probs, beliefs, expected_utilities)
				expected_pay_h += current_strategy[i] * expected_utilities[next_node]
				mu_h = infoset_belief.get(current_node)
				believed_infoset_action_utils[i] = believed_infoset_action_utils[i] + mu_h * expected_utilities[next_node]
			else:
				believed_infoset_action_utils[i] = np.array([-10.0, -10.0])

		believed_utility_I = np.matmul(current_strategy, believed_infoset_action_utils)
		player_id = current_node.player_id - 1

		# Compute the believed regrets of not playing each action at the infoset
		regrets = believed_infoset_action_utils[:, player_id] - believed_utility_I[player_id]
		current_infoset.regret_sum += regrets
		expected_utilities[current_node] = expected_pay_h
		return expected_pay_h

	def get_plausibility_order(self, strategy_profile):
		'''
		'''
		order = {}
		node_map = {}

		infoset_list = []
		for j in range(self.num_players):
			infoset_list += self.infosets[j]

		count = 0
		for infoset in infoset_list:
			for node in infoset.node_list:
				node_map[tuple(node.history)] = count
				count += 1

		for x in self.chance_map:
			node_map[tuple(x.history)] = count
			count += 1

		for t in self.terminal_nodes:
			node_map[tuple(t.history)] = count
			count += 1

		for infoset in infoset_list:
			infoset_strat = strategy_profile.get(infoset.infoset_id)

			for node in infoset.node_list:
				reachable_nodes = []
				unreachable_nodes = []
				h = tuple(node.history)

				# update order with strategy profile first
				for action in infoset_strat:
					ha = h + (action,)
					weight = infoset_strat[action]
					i_h = node_map.get(h)
					i_ha = node_map.get(ha)

					if weight > 0:
						order[(i_h, i_ha)] = 1
						order[(i_ha, i_h)] = 1
						reachable_nodes.append(ha)
					else:
						order[i_h, i_ha] = 1
						unreachable_nodes.append(ha)

				# update plausibility using transitive property
				for x in reachable_nodes:
					for y in reachable_nodes:
						i1 = node_map.get(x)
						i2 = node_map.get(y)
						order[(i1, i2)] = 1
						order[(i2, i1)] = 1

				for x in it.product(reachable_nodes, unreachable_nodes):
					i1 = node_map.get(x[0])
					i2 = node_map.get(x[1])
					order[(i1, i2)] = 1

		for x in self.chance_map:
			i_x = node_map.get(tuple(x.history))
			
			for e in self.chance_map[x]:
				i_xe = node_map.get(tuple(x.history) + (e,))
				order[(i_x, i_xe)] = 1
				order[(i_xe, i_x)] = 1

		return order, node_map

	def update_beliefs(self, strategy_profile, cur_beliefs):
		'''
		'''
		beliefs = {}

		order, node_map = self.get_plausibility_order(strategy_profile)
		trans_closure = self.get_transitive_closure(node_map, order)

		for j in range(self.num_players):
			for infoset in self.infosets[j]:

				if len(infoset.node_list) > 1:
					infoset_belief = {}
					total_reach_prob = 0.0
					for node in infoset.node_list:
						rp = self.compute_reach_prob(strategy_profile, node)
						total_reach_prob += rp
						infoset_belief[node] = rp

					if total_reach_prob == 0.0:

						if len(infoset.node_list) == 1:
							infoset_belief[node] = 1.0

						else:
							num_list = [node_map[tuple(x.history)] for x in infoset.node_list]
							infoset_closure = [x for x in trans_closure if x[0] in num_list and x[1] in num_list]
							prec_nodes = []

							for node in infoset.node_list:
								n = node_map[tuple(node.history)]
								prec_n = [x for x in infoset_closure if x[0] == n and x[1] != n]

								if len(prec_n) > 0:
									prec_nodes.append(node)
								else:
									not_present = [x for x in infoset_closure if x[0] == n or x[1] == n]
									if not_present == []:
										prec_nodes.append(node)

							for node in infoset.node_list:
								if node in prec_nodes:
									infoset_belief[node] = 1.0 / len(prec_nodes)
								else:
									infoset_belief[node] = 0.0

					else:
						num_list = [node_map[tuple(x.history)] for x in infoset.node_list]
						infoset_closure = [x for x in trans_closure if x[0] in num_list and x[1] in num_list]

						prec_nodes = []
						for node in infoset.node_list:
							n = node_map[tuple(node.history)]
							prec_n = [x for x in infoset_closure if x[0] == n and x[1] != n]
							if len(prec_n) > 0:

								prec_nodes.append(node)

						for h in infoset_belief:
							new_reach_prob = infoset_belief[h] / total_reach_prob
							infoset_belief[h] = new_reach_prob

					beliefs[infoset.infoset_id] = infoset_belief

				else:
					beliefs[infoset.infoset_id] = cur_beliefs.get(infoset.infoset_id)

		return beliefs

	def get_transitive_closure(self, node_map, order):
		'''
		'''
		tuple_list = [x for x in order if order[x] == 1]

		DG = nx.DiGraph(tuple_list)
		TC = nx.transitive_closure(DG, reflexive=None)
		
		return list(TC.edges())

	def compute_PBE(self, T):
		'''
		@arg (int) T: number of iterations

		Computes the Perfect Bayesian Equilibrium of a given game using
		the instant CFR (ICFR) method, modified to incorporate beliefs as
		well as strategy profiles over time
		'''
		print("called compute_PBE")
		beliefs = {}
		expected_utilities = {}

		# First, initialize key variables
		for j in range(self.num_players):
			infosets_j = self.infosets[j]

			for infoset in infosets_j:
				num_actions = len(infoset.action_space)
				infoset.regret_sum = np.zeros(num_actions)

				infoset.set_strategy(np.repeat(1.0 / num_actions, num_actions))
				infoset.strategy_sum = np.zeros(num_actions)
				infoset.strategy_sum_only = np.zeros(num_actions)
				infoset.reach_prob_sum = 0.0
				infoset.reach_prob = 0.0

				# beliefs will map information sets to probability distributions, 
				# which are a simplex over each node by history (each prob as float)
				infoset_id = infoset.infoset_id
				num_nodes = len(infoset.node_list)
				prob_dist = {}

				for n in infoset.node_list:
					prob_dist[n] = 1.0 / num_nodes
					expected_utilities[n] = np.zeros(self.num_players)

				beliefs[infoset_id] = prob_dist

		for chance_node in self.chance_map:
			expected_utilities[chance_node] = np.zeros(self.num_players)

		expected_val_cur_strategy = np.zeros(self.num_players)
		for t in range(T):
			if t % 100 == 0:
				print("t ", t)

			initial_reach_prob = {}
			for i in range(self.num_players + 1):
				initial_reach_prob[i] = 1.0

			expected_val_cur_strategy += self.recursive_pbe_cfr_helper(self.root, initial_reach_prob, beliefs, expected_utilities)
			for j in range(self.num_players):
				infosets_j = self.infosets[j]
				for infoset in infosets_j:
					infoset.update_strategy_pbe_cfr()

			strategy_tplus_1 = {}
			for j in range(self.num_players):
				for infoset in self.infosets[j]:
					strat = {}
					for i in range(len(infoset.action_space)):
						a = infoset.action_space[i]
						strat[a] = infoset.strategy[i]

					strategy_tplus_1[infoset.infoset_id] = strat

			beliefs = self.update_beliefs(strategy_tplus_1, beliefs)

		# computing average strategy and associated beliefs
		nash_strat = {}
		for j in range(self.num_players):
			for infoset in self.infosets[j]:
				nash_I = infoset.compute_average_strategy_pbe_cfr(T)
				num_actions = len(infoset.action_space)
				dist = {}
				for i in range(num_actions):
					a = infoset.action_space[i]
					dist[a] = nash_I[i]

				nash_strat[infoset.infoset_id] = dist

		beliefs = self.update_beliefs(nash_strat, beliefs)

		print("final nash_strat ")
		for x in nash_strat:
			print(x, nash_strat[x])
		print("\n")
		print("final beliefs ")
		for x in beliefs:
			print(x, beliefs[x])
		print("\n")
		
		return nash_strat, beliefs

	def verify_sequent_rat(self, pbe_assessment):
		'''
		'''
		strategy_profile = pbe_assessment[0]
		beliefs = pbe_assessment[1]

		regrets = {}

		for j in range(self.num_players):
			for infoset in self.infosets[j]:
				if len(infoset.action_space) > 1:
					max_regret = -100000.0
					iid = infoset.infoset_id
					infoset_belief = beliefs.get(iid)
					strategy_pay = 0.0
					for h in infoset.node_list:
						expected_pay = h.compute_pay(strategy_profile, self.chance_map, 1.0)[j]
						strategy_pay += infoset_belief.get(h) * expected_pay

					dummy_profile = strategy_profile.copy()

					for a in infoset.action_space:
						dummy_profile[iid] = {a: 1.0}
						action_pay = 0.0
						for h in infoset.node_list:
							expected_pay = h.compute_pay(dummy_profile, self.chance_map, 1.0)[j]
							action_pay += infoset_belief.get(h) * expected_pay

						infoset_regret = max(action_pay - strategy_pay, 0.0)
						if infoset_regret > max_regret:
							max_regret = infoset_regret

					regrets[iid] = max_regret

		return max(regrets.values())

