class BasePrior:
    def __init__(self):
        pass
    def logpdf(self,sample):
        raise NotImplementedError("{}.logpdf() must be implemented in classes inheriting from BasePrior".format(type(self).__name__))
    def rvs(self,n):
        raise NotImplementedError("{}.rvs() must be implemented in classes inheriting from BasePrior".format(type(self).__name__))
