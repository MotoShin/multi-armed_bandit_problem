import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class BernoulliArm():
	def __init__(self, p):
		self.p = p

	def draw(self):
		if self.p > random.random():
			return 1.0
		else:
			return 0.0

class EpsilonGreedy():
	def __init__(self, epsilon, counts, values):
		self.epsilon = epsilon
		self.counts = counts
		self.values = values

	def initialize(self, n_arms):
		self.counts = np.zeros(n_arms)
		self.values = np.zeros(n_arms)

	def select_arm(self):
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

def test_algorithm(algo, arms, num_sims, horizon):
	# 変数の初期化
	chosen_arms = np.zeros(num_sims * horizon)
	rewards = np.zeros(num_sims * horizon)
	cumulative_rewards = np.zeros(num_sims * horizon)
	sim_nums = np.zeros(num_sims * horizon)
	times = np.zeros(num_sims * horizon)

	for sim in range(num_sims):
		sim = sim + 1 # シミュレーション回数をカウント
		algo.initialize(len(arms)) # アルゴリズム設定を初期化

		for t in range(horizon):
			t = t + 1 # ラウンドの回数をカウント
			index = (sim - 1) * horizon + t - 1 # 現時点の回数（シミュレーション回数×ラウンド回数）をindexに代入
			sim_nums[index] = sim # simをsim_numsのindex番目の要素へ代入
			times[index] = t  # tをtimesのindex番目の要素へ代入

			chosen_arm = algo.select_arm() # select_arm()メソッドにより選択したアームをchosen_armへ代入
			chosen_arms[index] = chosen_arm # chosen_armをchosen_armsのindex番目の要素に代入
			reward = arms[chosen_arm].draw()
			rewards[index] = reward # rewardをrewardsのindex番目の要素へ代入

			if t == 1:
				# はじめてのラウンドならば
				# rewardをcumulative_rewardsのindex番目の要素へ代入
				cumulative_rewards[index] = reward
			else:
				cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

			algo.update(chosen_arm, reward)

	return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]

def main():
	theta = np.array([0.1, 0.1, 0.1, 0.1, 0.9]) # アームから報酬が得られる成功確率
	n_arms = len(theta)
	random.shuffle(theta)

	arms = map(lambda x: BernoulliArm(x), theta)
	arms = list(arms) # armsをリスト化する

	for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
		algo = EpsilonGreedy(epsilon, [], [])
		algo.initialize(n_arms)
		results = test_algorithm(algo, arms, 5000, 250) # シミュレーションの実行結果をresultsへ格納

		# プロット
		df = pd.DataFrame({"times": results[1], "rewards": results[3]})
		grouped = df["rewards"].groupby(df["times"])

		plt.plot(grouped.mean(), label="epsilon="+str(epsilon)) # 各ラウンドごとの報酬の平均値をεの値ごとにプロットする

	plt.legend(loc="best")
	plt.show()

main()
