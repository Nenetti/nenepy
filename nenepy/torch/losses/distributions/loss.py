import torch


class Loss:

    def __init__(self, p=None, reparameterization_trick=False):
        """

        Args:
            p (torch.distributions.Distribution or Loss):

        """
        if p is not None:
            p = self.to_loss(p)[0]

        self.p = p
        self.reparameterization_trick = reparameterization_trick

    def to_loss(self, *args):
        losses = [None] * len(args)
        for i, arg in enumerate(args):
            if isinstance(arg, Loss):
                losses[i] = arg
            elif isinstance(arg, torch.distributions.Distribution):
                losses[i] = Prob(arg)
            else:
                losses[i] = Value(arg)
        return losses

    def __add__(self, other):
        if isinstance(other, Loss):
            return Add(self, other)
        return Add(self, Loss(other))

    def __radd__(self, other):
        if isinstance(other, Loss):
            return Add(other, self)
        return Add(Loss(other), self)

    def __sub__(self, other):
        if isinstance(other, Loss):
            return Sub(self, other)
        return Sub(self, Loss(other))

    def __rsub__(self, other):
        if isinstance(other, Loss):
            return Sub(other, self)
        return Sub(Loss(other), self)

    def __mul__(self, other):
        if isinstance(other, Loss):
            return Mul(self, other)
        return Mul(self, Loss(other))

    def __rmul__(self, other):
        if isinstance(other, Loss):
            return Mul(other, self)
        return Mul(Loss(other), self)

    def __truediv__(self, other):
        if isinstance(other, Loss):
            return Div(self, other)
        return Div(self, Loss(other))

    def __rtruediv__(self, other):
        if isinstance(other, Loss):
            return Div(other, self)
        return Div(Loss(other), self)

    def __neg__(self):
        return Neg(self)

    def forward(self, *args, **kwargs):
        return self.p

    def expand(self, batch_shape, _instance=None):
        self.p.expand(batch_shape, _instance)

    @property
    def batch_shape(self):
        return self.p.batch_shape

    @property
    def event_shape(self):
        return self.p._event_shape

    @property
    def arg_constraints(self):
        return self.p.arg_constraints

    @property
    def support(self):
        return self.p.support

    @property
    def mean(self):
        return self.p.mean

    @property
    def variance(self):
        return self.p.variance

    @property
    def stddev(self):
        return self.p.variance.sqrt()

    def sample(self, sample_shape=torch.Size()):
        if self.reparameterization_trick:
            return self.rsample(sample_shape)
        else:
            with torch.no_grad():
                return self.p.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.p.rsample(sample_shape)

    def prob(self, value):
        return self.p.log_prob(value).exp()

    def log_prob(self, value):
        return self.p.log_prob(value)

    def cdf(self, value):
        return self.p.cdf(value)

    def icdf(self, value):
        return self.p.icdf(value)

    def enumerate_support(self, expand=True):
        return self.p.enumerate_support(expand)

    def entropy(self):
        return self.p.entropy()

    def perplexity(self):
        return torch.exp(self.p.entropy())

    def _extended_shape(self, sample_shape=torch.Size()):
        return self.p._extended_shape(sample_shape)

    def _validate_sample(self, value):
        return self.p._validate_sample(value)

    def _get_checked_instance(self, cls, _instance=None):
        return self.p._get_checked_instance(cls, _instance)

    def __call__(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:
            torch.Tensor or AbstractLoss:

        """
        return self.forward(*args, **kwargs)


class LossOperator(Loss):
    def __init__(self, p1, p2):
        """

        Args:
            p1 (Loss):
            p2 (Loss):

        """
        super().__init__()
        self.p1 = p1
        self.p2 = p2


class Add(LossOperator):

    def forward(self, *args, **kwargs):
        return self.p1() + self.p2()


class Sub(LossOperator):

    def forward(self, *args, **kwargs):
        return self.p1() - self.p2()


class Mul(LossOperator):

    def forward(self, *args, **kwargs):
        return self.p1() * self.p2()


class Div(LossOperator):

    def forward(self, *args, **kwargs):
        return self.p1() / self.p2()


class Neg(Loss):

    def forward(self, *args, **kwargs):
        return -self.p()


class Value(Loss):

    def __init__(self, p):
        super(Value, self).__init__()
        self.p = p

    def forward(self, *args, **kwargs):
        return self.p


class Prob(Loss):
    def __init__(self, p):
        super(Prob, self).__init__()
        if isinstance(p, torch.distributions.Normal):
            self.reparameterization_trick = True
        self.p = p

    def forward(self, x):
        return self.p.log_prob(x).exp()
