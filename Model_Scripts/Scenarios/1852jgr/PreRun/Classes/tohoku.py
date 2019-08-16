import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

import numpy as np
from scipy import stats

#tohokuKDE() makes a single KDE for a single column of x and y values
def tohokuKDE(onHeights, offHeights):
  #remove points where on or off shore heights are 0
  onHeights  =  onHeights[offHeights != 0.0]
  offHeights = offHeights[offHeights != 0.0]
  offHeights = offHeights[ onHeights != 0.0]
  onHeights  =  onHeights[ onHeights != 0.0]

  #remove points where on or off shore heights are >25.
  #this is purely for empirical reasons
#  onHeights  =  onHeights[offHeights < 25.0]
#  offHeights = offHeights[offHeights < 25.0]
  onHeights = onHeights[offHeights > 0.2]
  offHeights = offHeights[offHeights > 0.2]
  offHeights = offHeights[onHeights > 0.2]
  onHeights = onHeights[onHeights > 0.2]
  
  #build KDE
  values = np.vstack([onHeights, offHeights])
  kernel = stats.gaussian_kde(values,bw_method=0.25)

  return kernel;


#makeTohokuKDEs() builds a height KDE for
#each column of the amplification data
def makeTohokuKDEs(flNm='amplification_data.npy'):
  heightKdes     = []
  inundationKdes = []

  amplification_data = np.load(flNm);
  
  for d_idx in range(1,amplification_data.shape[1]):
    #height kdes
    onHeights   = amplification_data[:,0];
    offHeights  = amplification_data[:,d_idx]
    kernel      = tohokuKDE(onHeights, offHeights)
    heightKdes.append(kernel)

  return heightKdes


#plotTohokuKDEs() plots the height KDEs
def plotTohokuKDEs():
  heightKdes = makeTohokuKDEs('amplification_data.npy');
  
  xmin =  0.0;
  xmax =  max([ kde.dataset[0,:].max() for kde in heightKdes ]);#max([ val[:,0].max() for val in vals ]);#amplification_data[:,0].max()
  ymin =  0.0;
  ymax =  max([ kde.dataset[1,:].max() for kde in heightKdes ]);#max([ val[:,1].max() for val in vals ]);#amplification_data[:,1:].max()
  X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  positions = np.vstack([X.ravel(), Y.ravel()])

  for d_idx, kernel in enumerate(heightKdes):
    d = d_idx / 4.0;
    values = kernel.dataset.T;
    
    #make a plot
    #(see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
    fig, ax = plt.subplots()

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

    pltNm="tohoku_kde_d{:05.2f}".format(d)
    plt.savefig(pltNm+".png")
    plt.savefig(pltNm+".pdf")
    print("wrote: "+pltNm+".{png,pdf}")
    plt.close()
