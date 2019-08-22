import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

import numpy as np
#from scipy import stats
from AbstractKDE import AbstractKDE

#tohokuKDE() makes a single KDE for a single column of x and y values
def tohokuKDE(onHeights, offHeights, transformType='none', bw_method=0.25):
  #remove points where on or off shore heights are 0 (or less than 0)
  onHeights  =  onHeights[offHeights > 0.2]
  offHeights = offHeights[offHeights > 0.2]
  offHeights = offHeights[ onHeights > 0.2]
  onHeights  =  onHeights[ onHeights > 0.2]

  # #remove points where on or off shore heights are >25.
  # #this is purely for empirical reasons
  # onHeights  =  onHeights[offHeights < 25.0]
  # offHeights = offHeights[offHeights < 25.0]
  
  values = np.vstack([onHeights, offHeights])

  #print("min(values) is",np.min(values))
  #print("shape of values is",values.shape)
  #print("shape of np.sum(values,axis=0) is",np.sum(values,axis=0).shape)
  #print("shape of np.sum(values,axis=1) is",np.sum(values,axis=1).shape)

  ##transform if necessary
  #if transform == 'log': #lognormal
  #  values = np.log(values)

  #build KDE
  #kernel = stats.gaussian_kde(values,bw_method=bw_method)
  #kernel = AbstractKDE.AbstractKDE(values,bw_method=bw_method,transformType=transformType)
  kernel = AbstractKDE(values,bw_method=bw_method,transformType=transformType)

  return kernel;


#makeTohokuKDEs() builds a height KDE for
#each column of the amplification data
def makeTohokuKDEs(tohokuFile='amplification_data.npy', **kwargs):
  heightKdes     = []
  inundationKdes = []

  amplification_data = np.load(tohokuFile);
  
  for d_idx in range(1,amplification_data.shape[1]):
    #height kdes
    onHeights   = amplification_data[:,0];
    offHeights  = amplification_data[:,d_idx]
    kernel      = tohokuKDE(onHeights, offHeights, **kwargs)
    heightKdes.append(kernel)

  return heightKdes


#plotTohokuKDEs() plots the height KDEs
#def plotTohokuKDEs(tohokuFile='amplification_data.npy',outFolder='', **kwargs):
def plotTohokuKDEs(tohokuFile='amplification_data.npy',**kwargs):
  outFolder="."

  heightKdes = makeTohokuKDEs(tohokuFile, **kwargs);
  
  xmin =  0.01;
  xmax =  max([ kde.dataset[0,:].max() for kde in heightKdes ]);#max([ val[:,0].max() for val in vals ]);#amplification_data[:,0].max()
  ymin =  0.01;
  ymax =  max([ kde.dataset[1,:].max() for kde in heightKdes ]);#max([ val[:,1].max() for val in vals ]);#amplification_data[:,1:].max()
  X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
  positions = np.vstack([X.ravel(), Y.ravel()])

  for d_idx, kernel in enumerate(heightKdes):
    d = d_idx / 4.0;
    values = kernel.dataset.T;
    
    #make a plot
    #(see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
    fig, ax = plt.subplots(figsize=(8,6))

    #plot kde
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    #plot values
    ax.plot(values[:,0], values[:,1], 'k.', markersize=2);

    ax.set_xlim([xmin, xmax]);
    ax.set_ylim([ymin, ymax]);
    plt.xlabel("Onshore wave height (run up)"); 
    plt.ylabel("Offshore wave height (Geoclaw output)");
    plt.title("Tohoku data and KDE for d="+str(d));

    pltNm=outFolder+"/tohoku_kde_d{:05.2f}".format(d)
    plt.savefig(pltNm+".png",dpi=100)
    plt.savefig(pltNm+".pdf",dpi=100)
    print("wrote: "+pltNm+".{png,pdf}")
    plt.close()

#compute the conditional distribution associated with a kde
#(this means normalizing the kde across the horizontal direction)
def computeConditionalDistribution(kde,nOff=500,nOn=1000):
    offShoreHeights = np.linspace(0.0, kde.dataset[1, :].max(), num=nOff)
    onShoreHeights = np.linspace(0.0, kde.dataset[0, :].max(), num=nOn)
    wt = trapRuleWeights(onShoreHeights)

    #non-vectorized version. this was actually a little faster in tests
    # #these are the points at which we will evaluate the pdf
    # xy = np.zeros((2, len(onShoreHeights)))
    # xy[0, :] = onShoreHeights
    # condDist = np.zeros((len(offShoreHeights), len(onShoreHeights))) 
    # #loop across off-shore heights, compute kde, and normalize to get conditional distribution
    # for yidx, y in enumerate(offShoreHeights):
    #     xy[1, :] = y 
    #     kdeXY = kde.pdf(xy)
    #     nrm = sum(wt * kdeXY) # normalization factor
    #     if nrm > 0.: 
    #       kdeXY /= nrm        # normalize to get conditional distribution
    #     else:
    #       print("Warning: Conditional distribution is zero for off-shore height =",y)
    #     condDist[yidx, :] = kdeXY

    #vectorized version. this was actually a little slower in tests for some reason. 
    #but that seemed sort of fluky and maybe this scales better for larger datasets?
    xx,yy = np.meshgrid(onShoreHeights,offShoreHeights)
    xy=np.vstack([xx.ravel(), yy.ravel()])
    condDist = np.reshape(kde.pdf(xy), xx.shape)  #compute kdes
    nrm = np.matmul(condDist,wt)                  #compute integrals along rows
    condDist[nrm>0.,:] /= nrm[nrm>0.,None]        #normalize (row divide, nrm>0 avoids NaNs)

    return condDist, onShoreHeights, offShoreHeights

def plotConditionalDistribution(condDist,x,y,xmax=25.,ymax=25.,**kwargs):
    condDist = condDist[y <= ymax, :]
    condDist = condDist[:, x <= xmax]
    x = x[x <= xmax]
    y = y[y <= ymax]

    fig, ax = plt.subplots(**kwargs)

    X,Y = np.meshgrid(x,y)
    plt.contourf(X, Y, condDist)

    plt.xlabel('On shore')
    plt.ylabel('Off shore')
    #plt.xlim(0.,xmax)
    #plt.ylim(0.,ymax)
    plt.gca().set_aspect("equal")
    plt.colorbar()

def trapRuleWeights(x):
    """

    :param x:
    :return:
    """
    wt = np.zeros(len(x))
    wt[1:]  += 0.5*(x[1:]-x[:-1])
    wt[:-1] += 0.5*(x[1:]-x[:-1])
    return wt

#rescaled/recentered gauss-hermite quadrature points for 
#normal rv with mean mu and std sig
def gaussHermite(numPoints, mu=0, sig=1):
    #this version uses "physicists" hermite polynomials and is maybe a little trickier
    # x,w = np.polynomial.hermite.hermgauss( numPoints );
    # #normalize (makes sum(w)=1)
    # w /= np.sqrt(np.pi)
    # #recenter
    # x =  mu + np.sqrt(2)*sig*x

    #use "probabilists" hermite polynomials
    x,w = np.polynomial.hermite_e.hermegauss( numPoints );
    #normalize (makes sum(w)=1)
    w /= np.sqrt(2)*np.sqrt(np.pi)
    #recenter
    x =  mu + sig*x

    return x,w


