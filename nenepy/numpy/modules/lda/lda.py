import numpy as np
import torch
from scipy.special import psi
from torch.distributions import Categorical, Dirichlet
from tqdm import tqdm

EPS = np.finfo(np.float).eps


class LDA(object):
    """
    LDA (ギブスサンプリング)

    MLP(機械学習プロフェッショナルシリーズ)トピックモデル に準拠
    詳細は当該参考書 P71 参照

    各パラメータの詳細:

        パラメータ:

            D (int):                ドキュメント数
            V (int):                全ドキュメントの中で現れる単語の種類数
            K (int):                トピック数
            N (np.ndarray):         各ドキュメントの単語数
                Shape = N
            N_d (int):              ドキュメントdに含まれる単語数

            w (np.ndarray):         各ドキュメントの全単語の集合
                Shape = (D, N)
            w_d (np.ndarray):       ドキュメントdの全語群
                Shape = N
            w_dn (int):             ドキュメントdのn番目の単語

            z (list[np.ndarray]):   各ドキュメントの各単語のトピック
                Shape = (D, N)
            z_d (np.ndarray):       ドキュメントdの各単語のトピック
                Shape = N
            z_dn (int):             ドキュメントdのn番目の単語のトピック

            N_dk (np.ndarray):      ドキュメントdでトピックkが割り当てられた単語数

            N_k (np.ndarray):       ドキュメント集合全体でトピックkが割り当てられた単語数
            N_kv (np.ndarray):      ドキュメント集合全体でトピックkが割り当てられた語彙vの数

            θ_dk (np.ndarray):      ドキュメントdでトピックkが割り当てられる確率
            φ_kv (np.ndarray):      トピックkのとき語彙vが生成される確率


        ハイパーパラメータ:

            alpha(np.ndarray):  Shape = K   トピックの出現頻度の偏りを表すパラメータ
            beta(np.ndarray):   Shape = V   語彙の出現頻度の偏りを表すパラメータ

    """

    def __init__(self, data, alpha=0.1, beta=0.01, n_topic=100, is_train_hyperparameter=False, save_path="./result"):
        """

        Args:
            data (np.ndarray): numpy

        """

        self.D, self.V = data.shape
        self.K = n_topic
        self.N_d = data.sum(axis=1)

        self.w = self.bag_of_words_to_sentence(data=data)

        self.z = [np.full(shape=self.N_d[d], fill_value=-1, dtype=int) for d in range(self.D)]

        self.N_dk = np.zeros(shape=(self.D, self.K), dtype=int)
        self.N_k = np.zeros(shape=self.K, dtype=int)
        self.N_kv = np.zeros(shape=(self.K, self.V), dtype=int)

        self.theta = np.zeros(shape=(self.D, self.K))
        self.phi = np.zeros(shape=(self.K, self.V), dtype=int)

        self.alpha = np.full(shape=n_topic, fill_value=float(alpha))
        self.beta = np.full(shape=self.V, fill_value=float(beta))

        self.ari = []
        self.save_path = save_path
        self.train_mode = is_train_hyperparameter
        print(f"ドキュメント数: {self.D}, 語彙数: {self.V}, トピック数: {self.K}")

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    def gibbs_sampling(self, epoch):
        """
        ギブスサンプリング(メイン処理)

        Args:
            epoch(int): 試行回数

        """
        for _ in tqdm(range(epoch), ascii=True, leave=False):
            self._gibbs_sampling()

    def _gibbs_sampling(self):
        for d in range(self.D):
            for n in range(self.N_d[d]):
                w_dn = self.w[d][n]
                z_dn = self.z[d][n]

                if z_dn != -1:
                    self.N_dk[d][z_dn] -= 1
                    self.N_kv[z_dn][w_dn] -= 1
                    self.N_k[z_dn] -= 1

                self.theta[d] = self._calc_topic_probability(
                    alpha=self.alpha, beta=self.beta, N_dk=self.N_dk, N_kv=self.N_kv, N_k=self.N_k, w_dn=w_dn, d=d
                )

                updated_z_dn = self._sampling_topic_from_categorical(self.theta[d])

                self.N_dk[d][updated_z_dn] += 1
                self.N_kv[updated_z_dn][w_dn] += 1
                self.N_k[updated_z_dn] += 1

                self.z[d][n] = updated_z_dn

        if self.train_mode:
            self.alpha = self._update_alpha(alpha=self.alpha, D=self.D, N_dk=self.N_dk, N_d=self.N_d)
            self.beta = self._update_beta(beta=self.beta, K=self.K, N_kv=self.N_kv, N_k=self.N_k)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================
    @staticmethod
    def bag_of_words_to_sentence(data):
        """
        Bag-of-words形式のデータを、文書形式に変換する

        Args:
            data(np.ndarray):   処理するデータ (Bag-of-Words形式)

        Returns:
            np.ndarray: (D, N_d[d])

        """
        D, V = data.shape
        N_d = data.sum(axis=1)
        x = [np.zeros(shape=N_d[d], dtype=int) for d in range(D)]
        for d in range(D):
            x[d] = np.array([v for v in range(V) for i in range(data[d][v])])

        return x

    @staticmethod
    def _topic_to_bag_of_words(bow, z_dn, D, V):
        """
        Bag-of-words形式のデータを、文書形式に変換する

        Args:
            bow(np.ndarray):   処理するデータ (Bag-of-Words形式)

        """
        x = np.full(shape=(D, V), fill_value=0)
        for d in range(D):
            for i, n in enumerate(bow[d]):
                x[d, n] = z_dn[d][i]

        return x

    @staticmethod
    def _get_topics(w_dn, z_dn):
        pass

    @staticmethod
    def _calc_topic_probability(alpha, beta, N_dk, N_kv, N_k, w_dn, d):
        """
        トピックのサンプリング確率
        P(z_dn = k | W, Z/dn, α, β)を求める

        Args:
            alpha(np.ndarray):  トピックの出現頻度の偏りを表すパラメータ
            beta(np.ndarray):   語彙の出現頻度の偏りを表すパラメータ
            w_dn(int):          文書dのn番目の単語
            N_dk(np.ndarray):   文書dでトピックkが割り当てられた単語数
            N_k(np.ndarray):    文書集合全体でトピックkが割り当てられた単語数
            N_kv(np.ndarray):   文書集合全体で語彙vにトピックkが割り当てられた単語数
            d(int):             d番目の文書

        Returns:
            np.ndarray:         d番目の文書における、それぞれのトピックの確率

        """
        N_dk_dn = N_dk[d]
        N_kw_dn_dn = N_kv[:, w_dn]
        N_k_dn = N_k
        p = (N_dk_dn + alpha) * (N_kw_dn_dn + beta[w_dn]) / (N_k_dn + beta.sum())
        return p / p.sum()

    @staticmethod
    def _calc_theta(N_dk, alpha, N_d, K, D):
        return np.array([(N_dk[d] + alpha) / (N_d[d] + (alpha * K)) for d in range(D)])

    def calc_phi(self):
        return np.array([(self.N_kv[k] + self.beta) / (self.N_k[k] + (self.beta * self.V)) for k in range(self.K)])

    @classmethod
    def _update_alpha(cls, alpha, D, N_dk, N_d):
        """
        ハイパーパラメータ α を更新

        Args:
            alpha(np.ndarray):  トピックの出現頻度の偏りを表すパラメータ
            D(int):             文書数
            N_d(np.ndarray):    文書dに含まれる単語数
            N_dk(np.ndarray):   文書dでトピックkが割り当てられた単語数

        Returns:
            np.ndarray:         更新したハイパーパラメータ α

        """
        # ディガンマ関数に0をいれるとマイナス無限大に発散しWarningが出るが、式全体としては問題無いため警告を非表示にする

        with np.errstate(invalid='ignore'):
            a = np.array([cls._digamma(N_dk[d] + alpha) for d in range(D)]).sum(axis=0) - D * (cls._digamma(alpha))
            b = cls._digamma(N_d + alpha.sum()).sum() - D * cls._digamma(alpha.sum())

            # ライブラリの仕様上 Nan, Infinite, マイナス値 が発生する場合がある。これらは本来、値が0なので0に置き換える。
            # 原因は以下の2種類
            # 1. ディガンマ関数は0でマイナス無限大に発散するため、非常に小さい値を入れると Nan や Infinite が発生する
            # 2. N_dk[d]が0のとき計算式として0になるが、場合によって計算結果の誤差によってマイナス値が発生する (α は必ず正の実数値)
            a[(np.isnan(a)) | (np.isinf(a)) | (a < 0)] = 0
        return alpha * (a / b) + EPS

    @classmethod
    def _update_beta(cls, beta, K, N_kv, N_k):
        """
        ハイパーパラメータ β を更新

        Args:
            beta(np.ndarray):
            K(int):             トピック数
            N_k(np.ndarray):    文書集合全体でトピックkが割り当てられた単語数
            N_kv(np.ndarray):   文書集合全体でトピックkが割り当てられた語彙vの数

        Returns:
            np.ndarray:         更新したハイパーパラメータ β

        """
        # ディガンマ関数に0をいれるとマイナス無限大に発散しWarningが出るが、式全体としては問題無いため警告を非表示にする
        with np.errstate(invalid='ignore'):
            a = np.array([cls._digamma(N_kv[k] + beta) for k in range(K)]).sum(axis=0) - K * (cls._digamma(beta))
            b = cls._digamma(N_k + beta.sum()).sum() - K * cls._digamma(beta.sum())

            # ライブラリの仕様上 Nan, Infinite, マイナス値 が発生する場合がある。これらは本来、値が0なので0に置き換える。
            # 原因は以下の2種類
            # 1. ディガンマ関数は0でマイナス無限大に発散するため、非常に小さい値を入れると Nan や Infinite が発生する
            # 2. N_kv[k]が0のとき計算式として0になるが、場合によって計算結果の誤差によってマイナス値が発生する (β は必ず正の実数値)
            a[(np.isnan(a)) | (np.isinf(a)) | (a < 0)] = 0

        return beta * (a / b)

    @staticmethod
    def _sampling_topic_from_categorical(pi):
        """
        カテゴリカル分布(パラメータπ)からトピックをサンプリング

        Args:
            pi:     カテゴリカル分布のパラメータπ

        Returns:
            int:    サンプリングされたトピックのID

        """
        return Categorical(torch.from_numpy(pi)).sample().numpy().item()

    @staticmethod
    def dirichlet(alpha):
        return Dirichlet(torch.from_numpy(alpha)).sample((1000,)).mean(0).numpy()
        # return Dirichlet(torch.from_numpy(alpha)).sample().numpy()

    @staticmethod
    def _digamma(z):
        """
        ディガンマ関数(Ψ関数)(ガンマ関数の対数微分)の計算
        ディガンマ関数: Γ'(z)/Γ(z)

        Args:
            z(np.ndarray): 入力

        Returns:
            np.ndarray: 対数微分値

        """
        return psi(z)
