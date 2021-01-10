import numpy as np


class MutualInformation:

    @staticmethod
    def calc(p_x, p_y, p_xy):
        """

        Args:
            p_xy (np.ndarray):
            p_x (np.ndarray):
            p_y (np.ndarray):

        Returns:

        """
        N, C = p_xy.shape
        out = np.empty(shape=(N, C))
        for n in range(N):
            for c in range(C):
                out[n, c] = p_xy[n, c] * np.log2(p_xy[n, c] / (p_x[n] * p_y[c]))

        return out
        # N, C = p_xy.shape
        # out = np.empty(shape=(N, C))
        # for n in range(N):
        #     for c in range(C):
        #         out[n, c] = p_xy[n, c] * np.log(p_xy[n, c] / (p_x[n] * p_y[c]))
        #
        # return out

    @staticmethod
    def label_to_xy(label):
        """

        Args:
            label (np.ndarray):

        Returns:

        """
        n_sample, n_classes = label.shape
        for i in range(n_sample):
            indexes = np.where(label[i] == 1)[0]
            combinations = list(np.itertools.combinations(iterable=indexes, r=2))
            print(combinations)

    # @staticmethod
    # def calc(x, y):
    #     bins = len(x)
    #     p_xy, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    #
    #     p_x, _ = np.histogram(x, bins=xedges, density=True)
    #     p_y, _ = np.histogram(y, bins=yedges, density=True)
    #     p_x_y = p_x[:, np.newaxis] * p_y
    #
    #     dx = xedges[1] - xedges[0]
    #     dy = yedges[1] - yedges[0]
    #
    #     elem = p_xy * np.ma.log(p_xy / p_x_y)
    #
    #     return np.sum(elem * dx * dy), p_xy, p_x_y
