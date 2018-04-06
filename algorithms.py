import numpy as np
import random
import math

class Greedy():
	def __init__(self, counts, values):
		self.counts = counts
		self.values = values

	def initialize(self, n_arms):
		self.counts = np.zeros(n_arms)
		self.values = np.zeros(n_arms)

	def select_arm(self):
		if 0 in self.counts:
			index = np.where(self.counts==0)
			return index[0][0]
		else:
			return np.argmax(self.values)

	def update(self, chosen_arm, reward):
		self.counts[chosen_arm] = self.counts[chosen_arm] + 1
		n = self.counts[chosen_arm]

		value = self.values[chosen_arm]
		new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward
		self.values[chosen_arm] = new_value

class EpsilonGreedy():
	def __init__(self, epsilon, counts, values):
		self.epsilon = epsilon
		self.counts = counts
		self.values = values

	def initialize(self, n_arms):
		self.counts = np.zeros(n_arms)
		self.values = np.zeros(n_arms)

	def select_arm(self):
		if 0 in self.counts:
			index = np.where(self.counts==0)
			return index[0][0]
		else:
			if self.epsilon > random.random():
				# 確率εで探索を行う
				return np.random.randint(0, len(self.values))
			else:
				# 確率1-εで活用を行う
				return np.argmax(self.values)

	def update(self, chosen_arm, reward):
		self.counts[chosen_arm] = self.counts[chosen_arm] + 1 # アームを選んだ回数を更新
		n = self.counts[chosen_arm]

		value = self.values[chosen_arm]
		new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward
		self.values[chosen_arm] = new_value

class UCB():
	def __init__(self, counts, values):
		self.counts = counts
		self.values = values
		self.mu = [] # 期待値

	def initialize(self, n_arms):
		self.counts = np.zeros(n_arms)
		self.values = np.zeros(n_arms)
		self.mu = np.zeros(n_arms)

	def select_arm(self):
		n_arms = len(self.counts)
		if 0 in self.counts:
			index = np.where(self.counts==0)
			return index[0][0]
		else:
			ucb_values = np.zeros(n_arms)
			total_counts = np.sum(self.counts)
			for arm in range(n_arms):
				bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
				ucb_values[arm] = self.values[arm] + bonus
			return np.argmax(ucb_values)

	# TODO:UCBアルゴリズムに適した更新メソッドに変更
	def update(self, chosen_arm, reward):
		self.counts[chosen_arm] = self.counts[chosen_arm] + 1 # アームを選んだ回数を更新
		n = self.counts[chosen_arm]

		value = self.values[chosen_arm]
		new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward
		self.values[chosen_arm] = new_value
