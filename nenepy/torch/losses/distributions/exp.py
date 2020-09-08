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
        self.p, self.q = self.to_loss(p, q)

    def forward(self):
        samples = self.q.sample()
        return self.p(samples)
