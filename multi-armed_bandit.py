import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from algorithms import *

class BernoulliArm():
	def __init__(self, p):
		self.p = p

	def draw(self):
		if self.p > random.random():
			return 1.0
		else:
			return 0.0

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

		if sim % 100 == 0:
			print(sim)

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
	theta = np.array([0.2, 0.3, 0.4, 0.5]) # アームから報酬が得られる成功確率
	n_arms = len(theta)
	random.shuffle(theta)

	arms = map(lambda x: BernoulliArm(x), theta)
	arms = list(arms) # armsをリスト化する

	NUM_SIMS = 10000
	HORIZON = 10000

	#### Greedy ####
	algo_greedy = Greedy([], [])
	algo_greedy.initialize(n_arms)
	results = test_algorithm(algo_greedy, arms,  NUM_SIMS, HORIZON)

	df = pd.DataFrame({"times": results[1], "rewards": results[3]})
	grouped = df["rewards"].groupby(df["times"])

	plt.plot(grouped.mean(), label="Greedy") # 各ラウンドごとの報酬の平均値をεの値ごとにプロットする

	#### e-Greedy ####
	algo_egreedy = EpsilonGreedy(0.1, [], [])
	algo_egreedy.initialize(n_arms)
	results = test_algorithm(algo_egreedy, arms, NUM_SIMS, HORIZON)

	df = pd.DataFrame({"times": results[1], "rewards": results[3]})
	grouped = df["rewards"].groupby(df["times"])

	plt.plot(grouped.mean(), label="e-Greedy") # 各ラウンドごとの報酬の平均値をεの値ごとにプロットする

	#### UCB1 ####
	algo_ucb = UCB([], [])
	algo_ucb.initialize(n_arms)
	results = test_algorithm(algo_ucb, arms, NUM_SIMS, HORIZON)

	df = pd.DataFrame({"times": results[1], "rewards": results[3]})
	grouped = df["rewards"].groupby(df["times"])

	plt.plot(grouped.mean(), label="UCB1") # 各ラウンドごとの報酬の平均値をεの値ごとにプロットする

	plt.legend(loc="best")
	plt.show()

main()
