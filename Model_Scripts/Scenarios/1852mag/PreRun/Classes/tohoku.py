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


# #makeTohokuKDEs() builds a height KDE for
# #each column of the amplification data
# def makeTohokuKDEs(tohokuFile='amplification_data.npy', **kwargs):
#   heightKdes     = []
#   inundationKdes = []
# 
#   amplification_data = np.load(tohokuFile);
#   
#   for d_idx in range(1,amplification_data.shape[1]):
#     #height kdes
#     onHeights   = amplification_data[:,0];
#     offHeights  = amplification_data[:,d_idx]
#     kernel      = tohokuKDE(onHeights, offHeights, **kwargs)
#     heightKdes.append(kernel)
# 
#   return heightKdes


#makeTohokuKDEs() builds a KDE for
#each column of the amplification data
#if beta and n are None, then it's assumed that we're doing height KDEs and the data is not transformed
#if beta and n are not None, then it's assumed that we're doing inundation KDEs and the onShoreHeights are converted to inundation lengths
def makeTohokuKDEs(tohokuFile='amplification_data.npy', beta=None, n=None, **kwargs):
    kdes = []

    amplification_data = np.load(tohokuFile);
    onHeights   = amplification_data[:,0];

    #if beta and n are defined, convert to inundations
    if beta is not None and n is not None:
        onHeights = heightToInundation(onHeights, beta, n)
    elif beta is not None or n is not None:
        print("Warning: makeTohokuKDEs() called with only one of beta or n defined.")

    for d_idx in range(1,amplification_data.shape[1]):
        offHeights  = amplification_data[:,d_idx]
        kernel      = tohokuKDE(onHeights, offHeights, **kwargs)
        kdes.append(kernel)

    return kdes


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

def plotConditionalDistribution(condDist,x,y,xmax=25.,ymax=25.,outputFile=None,**kwargs):
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
    #plt.gca().set_aspect("equal")
    plt.colorbar()

    #print to file if one is specified
    if outputFile is not None:
        plt.savefig(outputFile)
        print("Wrote:",outputFile)

def makeTohokuConditionalDistributions(kdes,filePrefix="condDist_",distanceInterval=0.25,nOff=500,nOn=1000):
    """
    compute conditional distributions from kdes
    :param kdes: list of kde objects
    :return:
    """
  
    cdList    = []
    onHtList  = []
    offHtList = []

    #kernels are assumed to represent distances in intervals given by distanceInterval
    kdeDistances = np.arange(len(kdes)) * distanceInterval
    #loop over kernels
    for d_idx, kernel in enumerate(kdes):
        distance = kdeDistances[d_idx]
        print('Computing conditional distribution for kernel',d_idx)
        condDist, onShoreHeights, offShoreHeights = computeConditionalDistribution(kernel,nOff=nOff,nOn=nOn)
        fileName=filePrefix+str(d_idx)+'.npz'
        saveConditionalDistribution(fileName, condDist, onShoreHeights, offShoreHeights, distance, kernel)

        #append values to lists so we can return them
        cdList.append(condDist)
        onHtList.append(onShoreHeights)
        offHtList.append(offShoreHeights)

    return cdList, onHtList, offHtList, kdeDistances


#save a conditional distribution to an .npz file
def saveConditionalDistribution(fileName, condDist, onShoreHeights, offShoreHeights, distance, kde):
    np.savez(fileName, condDist=condDist, onShoreHeights=onShoreHeights, offShoreHeights=offShoreHeights, distance=distance, bw_method=kde.bw_method, transformType=kde.transformType)
    print('Saved:',fileName)

#read a conditional distribution from an .npz file written with makeTohokuConditionalDistributions()
def readConditionalDistribution(fileName):
    data = np.load(fileName)
    return data['condDist'], data['onShoreHeights'], data['offShoreHeights'], data['distance'], data['bw_method'].item(), data['transformType'].item()


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


def condDistFileName(prefix="",id=None,beta=None,n=None):
    #if beta and n are defined, do inundation
    if beta is not None and n is not None:
        fileName = "condDist_inun_b{:4.3f}_n{:4.3f}_".format(beta,n)
    #otherwise do height
    else:
        if beta is not None or n is not None:
            print("Warning: condDistFileName() called with only one of beta or n defined.")
        fileName = "condDist_ht_"

    fileName = prefix+fileName
    
    #if we have the id, assume that we want the full filename
    if id is not None:
        fileName = fileName+str(id)+".npz"

    return fileName
   

def heightToInundation(onHeights,beta,n):
    """

    :param onHeights:
    :param beta:
    :param n:
    :return:
    """
    #return np.power(onHeights,4/3) * 0.06 * np.cos(beta) / (n**2)
    return np.power(np.maximum(onHeights,0),4/3) * 0.06 * np.cos(np.pi*beta/180.0) / (n**2)


