"""
AbstractKDE Class: Standard Gaussian KDE functionality plus transform
Created By Justin Krometis
Created 08/16/2019
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class AbstractKDE:
    """
    This class handles a Gaussian KDE plus a transform
    """
    def __init__(self, values, bw_method=None, transformType='none'):
        """
        Initialize the class with priors
        :param values
        kde dataset
        """
        #rv_continuous.__init__(self)
        self.values = values
        self.dataset = np.atleast_2d(values)
        self.transformType = transformType

        #dimensions
        self.d, self.n = self.dataset.shape

        #if self.transform == 'log':
        #    self.kde = stats.gaussian_kde(np.log(values),bw_method=bw_method)
        #else:
        #    self.kde = stats.gaussian_kde(values,bw_method=bw_method)

        if self.transformType == 'log':
            self.transform   = np.log
            self.untransform = np.exp
        else:
            self.transform = lambda x: x #apparently this means the identity
            self.untransform = self.transform
        
        self.kde = stats.gaussian_kde(self.transform(values),bw_method=bw_method)

    #just defining this to align with stats.gaussian_kde
    #stats.gaussian_kde defines evaluate() and then aliases pdf() 
    #to it but this seems more clear
    def evaluate(self, samples):
        return self.pdf(samples)

    def logpdf(self, samples):
        """
        Calculate the logpdf
        :param samples:
        :return:
        """
        #if self.transform == 'log':
        #  lpdf = self.kde.logpdf( np.log(samples) ) - np.sum( np.log(samples), axis=0 )
        #else:
        #  lpdf = self.kde.logpdf( samples )
        if self.transformType == 'log':
            #lpdf = self.kde.logpdf( self.transform(samples) ) - np.sum( np.log(samples), axis=0 )
            #this should avoid cases where the sample includes zeros (which otherwise would produce -infs or divide by zero warnings)
            lpdf = np.zeros(samples.shape[-1])
            p = np.prod( samples, axis=0 )
            lpdf[p>0.]  = self.kde.logpdf( self.transform( samples[:,p>0.] ) ) - np.log(p[p>0.])
            lpdf[p<=0.] = np.ninf
        else:
            lpdf = self.kde.logpdf( samples )

        #print("lpdf is:")
        #print(lpdf)

        return lpdf

    #compute pdf at samples
    def pdf(self, samples):
        """
        Calculate the pdf
        :param samples:
        :return:
        """
        if self.transformType == 'log':
            #pdf = np.exp( self.logpdf( samples ) )
            #pdf = self.kde.pdf( self.transform( samples ) ) / np.prod( samples, axis=0 )
            #pdf = self.kde.pdf( self.transform( samples ) ) 
            #pdf[pdf > 0.] /= np.prod( samples[:,pdf>0.], axis=0 )
            
            #this should avoid cases where the sample includes zeros (which otherwise would produce -infs or divide by zero warnings)
            if self.d == 1:
              pdf = np.zeros(len(samples))
              pdf[samples>0.] = self.kde.pdf( self.transform( samples[samples>0.] ) ) / samples[samples>0.] 
            else:
              pdf = np.zeros(samples.shape[-1])
              p = np.prod( samples, axis=0 )
              pdf[p>0.] = self.kde.pdf( self.transform( samples[:,p>0.] ) ) / p[p>0.] 
        else:
            pdf = self.kde.pdf( samples )

        #print("pdf is:")
        #print(pdf)

        return pdf

    #draw random sample(s)
    def rvs(self, size=1):
        """
        Pick a random set of values
        :return:
        """

        #if self.transform == 'log':
        #    samples = np.exp(self.kde.resample(size))
        #else:
        #    samples = self.kde.resample(size)
        samples = self.untransform(self.kde.resample(size))

        return samples

    #plot the kde
    def plot(self, plotRange=None):
        #in 1d, plot kde vs. histogram
        if self.d == 1:
            xmin = self.dataset.min() if plotRange is None else plotRange[0]
            xmax = self.dataset.max() if plotRange is None else plotRange[1]
            x = np.linspace(xmin,xmax,num=100)
            
            #plt.hist(self.dataset,20,density=True,histtype='bar',label="dataset");
            #plt.plot(x,self.kde.pdf(x),label="KDE pdf");
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
            #plot untransformed
            ax1.hist(self.values,density=True,histtype='bar',label="Dataset");
            ax1.plot(x,self.pdf(x),label="KDE pdf");
            ax1.title.set_text("Untransformed space");
            ax1.legend();
            #plot transformed
            ax2.hist(self.transform(self.values),density=True,histtype='bar',label="Dataset");
            ax2.plot(self.transform(x),self.kde.pdf(self.transform(x)),label="KDE pdf");
            ax2.legend();
            ax2.title.set_text("Transformed space");

        #in 2d, plot kde vs. scatterplot of datapoints
        elif self.d == 2:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))

            #plot untransformed
            xmin =  self.dataset[0,:].min() if plotRange is None else plotRange[0]
            xmax =  self.dataset[0,:].max() if plotRange is None else plotRange[1]
            ymin =  self.dataset[1,:].min() if plotRange is None else plotRange[2]
            ymax =  self.dataset[1,:].max() if plotRange is None else plotRange[3]
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            xy = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(self.pdf(xy).T, X.shape)
            ax1.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
            ax1.plot(self.dataset[0,:], self.dataset[1,:], 'k.', markersize=2);
            ax1.set_xlim([xmin, xmax]);
            ax1.set_ylim([ymin, ymax]);
            ax1.set_aspect('auto')
            ax1.title.set_text("Untransformed space");
            #plot transformed
            xmin =  self.transform(self.dataset[0,:]).min()
            xmax =  self.transform(self.dataset[0,:]).max()
            ymin =  self.transform(self.dataset[1,:]).min()
            ymax =  self.transform(self.dataset[1,:]).max()
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            xy = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(self.kde.pdf(xy).T, X.shape)
            ax2.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
            ax2.plot(self.transform(self.dataset[0,:]), self.transform(self.dataset[1,:]), 'k.', markersize=2);
            ax2.set_xlim([xmin, xmax]);
            ax2.set_ylim([ymin, ymax]);
            ax2.set_aspect('auto')
            ax2.title.set_text("Transformed space");

        else:
            print("No method for plotting KDE of dimension",self.d,".")
        
    #allows calls like AbstractKDE(x) to return the pdf
    __call__ = pdf

