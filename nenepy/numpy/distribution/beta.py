# coding=utf-8
import math
import numpy as np

from scipy.stats import beta


class Beta(object):
    """
    ベータ分布の生成、条件付き確率を行うクラス
    """

    def __init__(self, a, b):
        """
        コンストラクター

        Args:
            a(float): パラメータ a
            b(float): パラメータ b
        """
        self.a = a
        self.b = b

    def get_sample(self):
        """
        ベータ分布から、変数を生成する

        Returns:
            float: 生成された値
        """
        return np.random.beta(self.a, self.b)

    def get_samples(self, k):
        """
        ベータ分布から 変数をk個生成する

        Args:
            k(int): 生成する変数の数

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = k)
        """
        return np.random.beta(self.a, self.b, k)

    def calc_probability(self, v):
        """
        入力された値が、この分布から生成される確率を求める

        Args:
            v(float): 評価対象の値

        Returns:
            float: この分布から生成された確率
        """
        return beta.pdf(v, self.a, self.b)
