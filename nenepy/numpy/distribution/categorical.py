# coding=utf-8
import numpy as np
from scipy.stats import dirichlet

from .multinomial import Multinomial


class Categorical(object):
    """
    カテゴリカル分布の生成、条件付き確率を行うクラス
    ただしカテゴリカル分布は多項分布の試行回数 1回の時の分布なので、多項分布を呼び出すだけ
    """

    def __init__(self, pi):
        """
        コンストラクター

        Args:
            pi(np.ndarray): パラメータ π
        """
        self.multinomial = Multinomial(pi=pi, times=1)

    @staticmethod
    def sampling(pi):
        """
        多項分布から、試行回数1回でサンプリングする

        Args:
            pi(np.ndarray): パラメータ π

        Returns:
            np.ndarray: one-hot-vector(次元数 = piの次元数)
        """
        return Multinomial(pi=pi, times=1).get_sample()
