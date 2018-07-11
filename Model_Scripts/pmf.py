import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats



class PMF:
    """A class containing data for a discrete probability mass function"""
    def __init__(self, values, probabilities):

        # check for correct types of input
        if type(values) != type(np.array([])):
            if type(values) == type([]):
                values = np.array(values)
            else:
                raise ValueError('values must be of type numpy.ndarray or list')
        if type(probabilities) != type(np.array([])):
            if type(probabilities) == type([]):
                 probabilities = np.array(probabilities)
            else:
                raise ValueError('probabilities must be of type numpy.ndarray or list')

        # check for correct dimensions of input
        if len(values) != len(probabilities):
            raise ValueError("values and probabilities must have the same length")
        if not all(i >= 0 for i in probabilities):
            raise ValueError("Probabilities must be positive")

        # normalize probabilities
        if np.sum(probabilities) != 1:
            probabilities /= np.sum(probabilities)

        # save data
        self.vals = values
        self.probs = probabilities

    def mean(self):
        """Returns the mean of the PMF"""
        return np.dot(self.vals, self.probs)/sum(self.probs)

    def draw(self, n=1):
        """Returns a list of n draws from the distribution"""
        return np.random.choice(self.vals, n, p=self.probs)

    def display(self):
        """Displays the pmf"""
        width = abs(self.vals[1] - self.vals[0])
        print('width', width)
        plt.bar(self.vals, self.probs, align='center', width=width)
        plt.show()

    def integrate(self, distrb):
        """Integrates this distribution with the one passed in"""

        """# integrate with another PMF
        if isinstance(distrb, PMF):
            total = 0
            for i, val in enumerate(self.vals):
                if val in distrb.vals:
                    total += self.probs[i]*distrb.probs[distrb.vals.index(val)]
            return total"""

        # integrate with a normal distribution from stats.norm
        if type(distrb) == type(stats.norm()):
            total = 0
            for i, val in enumerate(self.vals):
                total += self.probs[i]*distrb.pdf(val)
            return total
        elif type(distrb) == type(stats.skewnorm()):
            total = 0
            for i, val in enumerate(self.vals):
                total += self.probs[i]*distrb.pdf(val)
            return total
        elif type(distrb) == type(stats.chi()):
            total = 0
            for i, val in enumerate(self.vals):
                total += self.probs[i]*distrb.pdf(val)
            return total
        else:
            raise ValueError("The distribution that you passed in has not been implemented")


class PMFData:
    def __init__(self, row_header, col_header, data):

        # check input has correct dimensions
        if len(row_header) != data.shape[0] or len(col_header) != data.shape[1]:
            raise ValueError("Shapes of data do not align")

        # save input
        self.col_header = col_header
        self.row_header = row_header
        self.data = data
        self.m, self.n = data.shape

    def getPMF(self, d, y, tol=0.4):
        # find the column in data most closesly associated with the input value d
        # Ex. find the recorded distance from shore closest to the requested distance d
        closest_idx = min(range(self.n), key = lambda i: abs(self.col_header[i] - d))
        column = self.data[:,closest_idx]

        # from the predetermined column, select all entries of the row header where the column value is within tol of y
        # Ex. make a list of the run-up heights that are associated with wave heights within tol of the given wave height y
        # for now we are doing this by saying the tolerance tol is the default value (2m) onshore (d=0), but increases with distance d
        # from shore, as we are dealing with water depth which has an order of magnitude change for d>0
        tol = .25*y #this is completely arbitrary and should be checked in future studies...!!!
        values = [self.row_header[i] for i in range(self.m) if abs(column[i] - y) < tol]

        # bin the values to obtain data for a PMF
        probabilities, vals = np.histogram(values, density=True)

        # return a PMF object
        return PMF(vals[:-1], probabilities)
    #def get Inundation(self, d, y)
        #find the inundation from on-shore wave heights
        #inun_values = np.power(self.getPMF(d, y).vals, 4/3)*0.06*cos()
        #inun_probability = 
if __name__ == "__main__":
    # testing on a version of 'amplification_data.npy'
    data = np.load('amplification_data.npy')
    row_header = data[:,0]
    col_header = np.arange(len(data[0]) - 1)/4


    p = PMFData(row_header, col_header, data[:,1:])
    pmf = p.getPMF(4.75,3)
    pmf.display()
    pmf.mean()
    pmf.integrate(stats.norm(6,1))
