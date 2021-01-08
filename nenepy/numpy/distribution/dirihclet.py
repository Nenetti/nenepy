# coding=utf-8
import numpy as np
from scipy.stats import dirichlet


class Dirichlet(object):
    """
    ディリクレ分布の生成、条件付き確率を行うクラス
    """

    def __init__(self, alpha):
        """
        コンストラクター

        Args:
            alpha(np.ndarray): パラメータ α
        """
        self.alpha = alpha

    @staticmethod
    def sampling(alpha, k):
        """
        ディリクレ分布から alphaの次元数と同じ次元数を持つベクトルを、k個生成する

        Args:
            alpha(np.ndarray): パラメータ α
            k(int): 生成するベクトルの数

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = k * alphaの次元数)
        """
        samples = [np.random.dirichlet(alpha) for i in range(k)]
        return np.array(samples)

    def get_sample(self):
        """
        ディリクレ分布から alphaの次元数と同じ次元数を持つベクトルを生成する
        生成されるベクトルの総和は必ず 1

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = alphaの次元数)
        """
        return np.random.dirichlet(self.alpha)

    def get_samples(self, k):
        """
        ディリクレ分布から alphaの次元数と同じ次元数を持つベクトルを、k個生成する

        Args:
            k(int): 生成するベクトルの数

        Returns:
            np.ndarray: 生成されたベクトル(次元数 = k * alphaの次元数)
        """
        samples = [self.get_sample() for i in range(k)]
        return np.array(samples)

    def calc_probability(self, vector):
        """
        入力されたベクトルが、この分布から生成される確率を求める

        Args:
            vector(np.ndarray): 評価対象のベクトル

        Returns:
            float: この分布から生成された確率
        """
        return dirichlet.pdf(vector, self.alpha)
