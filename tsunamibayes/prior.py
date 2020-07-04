import scipy.stats as stats

class BasePrior:
    def __init__(self):
        raise NotImplementedError("__init__() must be implemented in classes inheriting from BasePrior")
    def logpdf(self,sample):
        raise NotImplementedError("logpdf() must be implemented in classes inheriting from BasePrior")
    def rvs(self,n=1):
        raise NotImplementedError("rvs() may be implemented in classes inheriting from BasePrior")

class TestPrior(BasePrior):
    def __init__(self):
        0
    def logpdf(self,sample):
        return stats.halfnorm.logpdf(sample['magnitude'])
    def rvs(self):
        return [stats.halfnorm.rvs()]
