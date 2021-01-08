# coding=utf-8
import math
from functools import reduce

import numpy as np
from operator import mul


class Multinomial(object):
    """
    多項分布の生成、条件付き確率を行うクラス
    """

    def __init__(self, pi, times):
        """
        コンストラクター

        Args:
            pi(np.ndarray): パラメータ π (それぞれの出やすさ)
            times(int): パラメータ M (試行回数)
        """
        self.pi = pi
        self.times = times

    @staticmethod
    def sampling(pi, k):
        """
        多項分布から、piの次元数と同じ次元数を持つベクトルを生成する

        Args:
            pi(np.ndarray): パラメータ π (それぞれの出やすさ)
            k(int): パラメータ M (試行回数)

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = alphaの次元数)
        """
        return np.random.multinomial(k, pi)

    def get_sample(self):
        """
        多項分布から、piの次元数と同じ次元数を持つベクトルを生成する

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = alphaの次元数)
        """
        return np.random.multinomial(self.times, self.pi)

    def get_samples(self):
        """
        多項分布から piの次元数と同じ次元数を持つベクトルを M個生成する

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = k * alphaの次元数)
        """
        return np.random.multinomial(self.times, self.pi, self.times)

    def calc_probability(self, vector):
        """
        入力されたベクトルが、この分布から生成される確率を求める

        Args:
            vector(np.ndarray): 評価対象のベクトル

        Returns:
            float: この分布から生成された確率
        """
        cumulative_array = np.array([((pi_k ** m_k) / math.factorial(m_k)) for (pi_k, m_k) in zip(self.pi, vector)])
        cumulative_sum = reduce(mul, cumulative_array)
        return math.factorial(self.times) * cumulative_sum
