import scipy.stats as stats

class BasePrior:
    """A base class to define the necessary functions to construct a prior
    distribution.
    """
    def __init__(self):
        raise NotImplementedError(
            "__init__() must be implemented in classes "
            "inheriting from BasePrior"
        )
    def logpdf(self,sample):
        raise NotImplementedError(
            "logpdf() must be implemented in classes inheriting from BasePrior"
        )
    def rvs(self,n=1):
        raise NotImplementedError(
            "rvs() may be implemented in classes inheriting from BasePrior"
        )

class TestPrior(BasePrior):
    def __init__(self):
        0
    def logpdf(self,sample):
        """Computes the log of the probability density function for the
        sample's magnitude.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float value associated with the
            sample's magnitude.

        Returns
        -------
        logpdf : float
            The log of the probability density function for the sample.
        """
        return stats.halfnorm.logpdf(sample['magnitude'])

    def rvs(self):
        """Returns a float value for a halfnormal random variate."""
        return [stats.halfnorm.rvs()]
