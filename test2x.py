import itertools
import pprint
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNN

from nenepy.numpy.modules import MutualInformation
from nenepy.numpy.modules.lda import LDA

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=10000)
np.set_printoptions(formatter={'int': '{:3d}'.format})


def tes(n, n2):
    # path = Path(f"{Path.home()}/datasets/SIGVerse/trial3_phrase/noisy_label/label_{n2}.csv")
    # class_list = np.loadtxt(f"{Path.home()}/datasets/SIGVerse/trial3_phrase/noisy_class_list/class_list.txt", delimiter=',', dtype=str)[1:]
    path = Path(f"{Path.home()}/datasets/SIGVerse/trial3_noise/noisy_label/label_{n}_{n2}_2.csv")
    class_list = np.loadtxt(f"{Path.home()}/datasets/SIGVerse/trial3_noise/noisy_class_list/class_list_{n}.txt", delimiter=',', dtype=str)[1:]
    labels = np.genfromtxt(path, delimiter=',').astype(np.int)
    labels = labels[:, 1:]
    lda = LDA(labels, alpha=0.1, beta=0.01, is_train_hyperparameter=True, n_topic=20)
    lda.gibbs_sampling(50)
    n_v = lda.N_kv.transpose((1, 0)).sum(axis=1)
    for i, c in enumerate(class_list):
        print(lda.N_kv.transpose((1, 0))[i], lda.N_kv.transpose((1, 0))[i].sum(), c)

    # Phi = φ_kv = トピックkに単語vが割り当てられる確率
    phi = lda.calc_phi()
    C, V = phi.shape

    # phi = phi / phi.sum(0)
    # phi = phi / n_v
    phi = phi.transpose((1, 0))
    print(phi)

    p_x = phi
    p_y = LDA.dirichlet(lda.alpha)
    p_xy = p_x * p_y
    p_x = p_xy.sum(axis=1)
    print(p_x.shape, p_y.shape, p_xy.shape)
    mi = MutualInformation.calc(p_x, p_y, p_xy).sum(axis=1)
    # mi /= mi.max()
    t = lda.N_kv
    for i in range(20):
        print(mi[i], class_list[i])
    print()
    for i in range(20, V):
        print(mi[i], class_list[i])

    print(t.shape)
    sum_v = t.sum(axis=0)
    sum_v[sum_v == 0] = 1
    with open(f"./N_{path.name}", "w") as f:
        line = class_list
        line = ",".join(line)
        f.write(f",{line}\n")

        for i in range(len(t)):
            print(i, t.shape, sum_v.shape)
            # sum_v = t[i].sum()
            # if sum_v == 0:
            #     sum_v = 1
            line = [f"Topic{i}"] + [f"{(k * 100) / sum_v[m]:.0f}" if k != 0 else f"" for m, k in enumerate(t[i])]
            line = ",".join(line)
            print(len(t), line)
            f.write(f"{line}\n")

        line = [f"{t[:, i].sum():.1f}" for i in range(t.shape[1])]
        line = ",".join(line)
        f.write(f"\n,{line}\n")

    # with open(f"./N_{path.name}", "w") as f:
    #     for i in range(len(t)):
    #         sum_v = t[i].sum()
    #         line = [class_list[i]] + [f"{k:.1f}" if k != 0 else f"" for k in t[i] * 100 / sum_v] + [f"{sum_v}"]
    #         line = ",".join(line)
    #         print(len(t), line)
    #         f.write(f"{line}\n")

    # np.savetxt(f"./N_{path.name}", t)
    np.savetxt(f"./MI_{path.name}", mi, fmt='%.5f')
    #
    # sys.exit()
    #
    # # print(phi.shape)
    # # phi = lda.calc_phi().sum(axis=0)
    # # print(phi)
    # # print()
    # # t = (lda.calc_phi().transpose((1, 0)) * LDA.dirichlet(lda.alpha)).transpose(1, 0) / lda.calc_phi().sum(axis=0)
    # # print(t)
    #
    # # pprint.pprint(list(zip(lda.alpha, LDA.dirichlet(lda.alpha))))
    # # print()
    # # print(lda.N_kv.transpose((1, 0)))
    #
    # # diri = LDA.dirichlet(lda.alpha)
    # # print(phi.shape)
    # # print(phi.transpose(1, 0))
    # # print(np.sum(diri * phi, axis=1))
    # # print(phi.sum(axis=0), phi.sum(axis=0).shape)
    # # print(np.argmax(lda.calc_phi(), axis=0))
    # # print(np.sort(p_y)[::-1])
    # print(p_y)
    # print()
    #
    # p_xy = phi * p_y
    # # print(p_xy)
    #
    # # print(p_xy)
    # p_x = np.sum(p_xy, axis=1)
    # # p_x = np.ones_like(p_x)
    # print(p_x)
    # print()
    # # print(p_x.shape, p_y.shape, p_xy.shape)
    # # print(p_y)
    #
    # mi = MutualInformation.calc(p_x, p_y, p_xy)
    # print(mi)
    # for i, c in enumerate(class_list):
    #     t = mi.sum(axis=1)[i]
    #     print(t, c, t < 0.02)
    # # pprint.pprint(.tolist())
    # # for i, c in enumerate(class_list):
    # #     print(mi.mean(axis=1)[i], c)
    #
    # sys.exit()


if __name__ == '__main__':
    for n in [4, 8, 12, 16, 20]:
        for n2 in [20, 40, 60, 80, 100]:
            tes(n, n2)
