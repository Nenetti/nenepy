# import torch
# from nenepy.torch.losses.distributions import KullbackLeibler
# from nenepy.torch.losses.distributions import E
# from nenepy.torch.losses.distributions import Log
# from torch.distributions import Normal, Bernoulli
# import numpy as np
#
# loc = np.zeros(shape=10, dtype=np.float32)
# scale = np.ones(shape=10, dtype=np.float32)
#
# # loc2 = np.random.uniform(0, 1, size=10)
# # scale2 = np.random.uniform(0, 1, size=10)
# loc2 = np.ones(shape=10, dtype=np.float32)
# scale2 = np.ones(shape=10, dtype=np.float32)
#
# p = Normal(loc=torch.from_numpy(loc), scale=torch.from_numpy(scale))
# q = Normal(loc=torch.from_numpy(loc2), scale=torch.from_numpy(scale2))
# p1 = Bernoulli(torch.from_numpy(np.random.uniform(0, 1, (3, 256, 256))))
# p2 = Bernoulli(torch.from_numpy(np.random.uniform(0, 1, (3, 256, 256))))
# reconst = E(q=p2, p=Log(p1))
# # print(reconst())
#
# a = torch.ones((2, 3, 4, 5, 6))
# print(torch.mean(a, dim=[1,2]))
# # loss = (KullbackLeibler(p, q) - E(q=q, p=Log(p)))
# # loss = (kl * 5) - 1 + 2
# # loss = -kl - reconst
#
# # print(loss())
# # e1 = torch.exp(e)
# # e2 = torch.log(e1)
# # print(kl)
# # print(e1)
# # print(e2)
# # print(kl)
import torch

from nenepy.torch.nn.modules import DynamicUpsample
from nenepy.utils import Timer


module = DynamicUpsample(scale_factor=(4, 4))
x = torch.ones((2, 3, 56, 56))
print(str(module.__class__).split(".")[-1].split("'")[0])
# t = Timer()
# for i in range(1000):
#     y = module(x, scale_factor=(4, 4))
#
# print(t, y.shape)
