# 多腕バンディット問題を３つのアルゴリズムで比較

greedyアルゴリズムとε-greedyアルゴリズムとUCB1アルゴリズムによって多腕バンディット問題を解き、その結果を比較することを目的としたプログラムである。

# 動作環境

- PC
	- MacBook Pro (Retina, 13-inch, Late 2013)
- OS
	- macOS Sierra バージョン 10.12.6
- メモリ
	- 8 GB 1600 MHz DDR3
- 言語
	- Python 3.5.2
	- Anaconda custom (64-bit)
- 使用ライブラリ
	- pandas
	- numpy
	- matplotlib
	- random
	- math

# 実行方法

以下のコマンドを実行する。

`python multi-armed_bandit.py`

# 結果

プログラム実行結果は以下のようになった。

なお、縦軸が時間辺りの平均報酬で横軸がプレイ回数である。

![実行結果](https://raw.githubusercontent.com/MotoShin/multi-armed_bandit_problem/images/figure_1.png)

# 考察

プログラム実行結果よりこの多腕バンディット問題ではUCB1、ε-greedy、greedyの順で良い結果になることがわかった。
