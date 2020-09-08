import torch
from nenepy.torch.losses.distributions import Loss


class Expectation(Loss):

    def __init__(self, p, q):
        """

        Args:
            p (Loss):
            q (Loss):

        """
        super(Expectation, self).__init__()
        p, q = self.to_loss(p, q)
        self.p = p
        self.q = q

    def forward(self):
        samples = self.q.sample()
        return self.p(samples)
