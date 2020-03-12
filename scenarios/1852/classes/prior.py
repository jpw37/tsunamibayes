# TODO: GET ACCESS TO BASE CLASS
from prior import BasePrior

class Prior_1852(BasePrior):
    def __init__(self):
        raise NotImplementedError

    def logpdf(self):
        raise NotImplementedError

    def rvs(self):
        raise NotImplementedError