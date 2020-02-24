import matplotlib
matplotlib.use('agg', warn=False, force=True)
from matplotlib import pyplot as plt

import numpy as np
from Gauge import from_json
from tohoku import tohokuKDE, computeConditionalDistribution, readConditionalDistribution, saveConditionalDistribution, condDistFileName

gauges_path = "InputData/gauges.npy"
amp_data_path = "../InputData/amplification_data.npy"
cond_dist_path = "condDist_"
height_llh_path = "../InputData/gaugeHeightLikelihood.npy"
inun_llh_path = "../InputData/gaugeInundationLikelihood.npy"
gauge_output = 'InputData/gauge_llh_ht.png'


def buildGaugeLikelihoods(gaugeFile=gauges_path, condDistFilePrefix=cond_dist_path, tohokuFile=amp_data_path, heightFile=height_llh_path, inundationFile=inun_llh_path, gaugeIds=-1, distanceInterval=0.25):
    """

    :param gaugeFile:
    :param tohokuFile:
    :param heightFile:
    :param inundationFile:
    :param offshorePoints:
    :param gaugeIds:
    :return:
    """

    #Load gauges
    print("Loading gauges from " + gaugeFile)
    gauges = list(np.load(gaugeFile, allow_pickle=True))
    gauges = [from_json(gauge) for gauge in gauges]

    if gaugeIds != -1:
        gauges = [gauges[i] for i in gaugeIds]

    amplification_data = np.load(tohokuFile)

    kdeDistances = np.arange(amplification_data.shape[1]) * distanceInterval

    # These are the values that we will fill in
    heightLikelihood     = [] #np.zeros((len(offShoreHeights), len(gauges)))
    inundationLikelihood = [] #np.zeros((len(offShoreHeights), len(gauges)))

    print("Building gauge likelihoods...")
    for gid, gauge in enumerate(gauges):
        d_idx = np.argmin(np.abs(kdeDistances - gauge.distance))
        #cdFile = condDistFilePrefix+str(d_idx)+".npz"
        #print("Gauge",gid,": Reading conditional distribution from",cdFile)
        #condDist, onShoreHeights, offShoreHeights, distance, bw_method, transformType = readConditionalDistribution(cdFile)
        #if distance != gauge.distance:
        #    print("Warning: Gauge distance"+str(gauge.distance)+"does not match conditional distribution distance"+str(distance))

        #compute height likelihood
        if (gauge.kind[1]):
            #read conditional distribution
            cdFile = condDistFileName(condDistFilePrefix,d_idx)
            print("Gauge",gid," (Height): Reading conditional distribution from",cdFile)
            condDist, onShoreHeights, offShoreHeights, distance, bw_method, transformType = readConditionalDistribution(cdFile)
            if distance != gauge.distance:
                print("Warning: Gauge distance"+str(gauge.distance)+"does not match conditional distribution distance"+str(distance))

            wt = trapRuleWeights(onShoreHeights);  # trapezoidal rule
            heightLikelihood.append( computeLikelihoodPdf(condDist, gauge.height_dist.pdf, onShoreHeights, wt, offShoreHeights) )
        else:
            heightLikelihood.append( np.zeros( offShoreHeights.shape ) )

        # #inundation is different: the conditional distribution has to be computed using gauge information
        #compute inundation likelihood
        if (gauge.kind[2]):
            ##compute kde
            #print('Computing kde for inundation gauge',gid)
            #onHeights = amplification_data[:, 0]
            #offHeights = amplification_data[:, d_idx+1]
            #inundations = heightToInundation(onHeights, gauge)
            #kernel = tohokuKDE(inundations, offHeights, transformType=transformType, bw_method=bw_method)
            #
            ##compute conditional distribution and save it
            #print('Computing conditional distribution for inundation gauge',gid)
            #condDist, inundations, offShoreHeights = computeConditionalDistribution(kernel,nOn=5000)
            #fileName=condDistFilePrefix+'inun_' + str(gid) + '.npz'
            #saveConditionalDistribution(fileName, condDist, inundations, offShoreHeights, distance, kernel)

            #read conditional distribution
            cdFile = condDistFileName(condDistFilePrefix,d_idx, gauge.beta, gauge.n)
            print("Gauge",gid," (Inundation): Reading conditional distribution from",cdFile)
            condDist, inundations, offShoreHeights, distance, bw_method, transformType = readConditionalDistribution(cdFile)
            if distance != gauge.distance:
                print("Warning: Gauge distance"+str(gauge.distance)+"does not match conditional distribution distance"+str(distance))

            #now compute likelihood
            wt = trapRuleWeights(inundations);  # trapezoidal rule
            inundationLikelihood.append( computeLikelihoodPdf(condDist, gauge.inundation_dist.pdf, inundations, wt, offShoreHeights) )
        else:
            inundationLikelihood.append(np.zeros( offShoreHeights.shape ) )  #appending zeros for gauges without inundation

    outputData = np.insert(np.asarray(heightLikelihood).T, 0, offShoreHeights, axis=1)
    np.save(heightFile, outputData)

    outputData = np.insert(np.asarray(inundationLikelihood).T, 0, offShoreHeights, axis=1)
    np.save(inundationFile, outputData)


def computeLikelihoodPdf(condDist, gaugePdf, x, wt, offShoreHeights):
    """

    :param kdePdf:
    :param gaugePdf:
    :param x:
    :param wt:
    :param offShoreHeights:
    :return:
    """
    gPdf = gaugePdf(x)
    # test quadrature rule:
    intGaugePdf = sum(wt * gPdf)
    #print("intGaugePdf = ",intGaugePdf)
    if np.abs(intGaugePdf - 1.0) > 0.05:
        print("WARNING: Integration rule may not be accurate enough. Integration of gauge PDF yielded: {:.6f}".format(
            intGaugePdf))

    #integrate conditional distribution horizontally (across on-shore height or inundation) against gauge pdf
    likelihood = np.matmul(condDist,wt * gPdf)

    #integrate vertically and normalize
    intLikelihood = sum(likelihood * trapRuleWeights(offShoreHeights))
    likelihood /= intLikelihood

    return likelihood


# makeInundationKDEs() builds an inundation KDE for each gauge
def makeInundationKDEs(gauges, tohokuFile='amplification_data.npy', distanceInterval=0.25, **kwargs):
    """

    :param gauges:
    :param tohokuFile:
    :return:
    """
    inundationKdes = []

    amplification_data = np.load(tohokuFile)

    for gid, gauge in enumerate(gauges):
        # figure out which kde to use for this gauge
        kdeDistances = np.arange(amplification_data.shape[1]) * distanceInterval
        d_idx = np.argmin(np.abs(kdeDistances - gauge.distance)) + 1
        print("Making an inundation KDE for gauge",gid,"using amplification data column",d_idx)

        # inundation kdes
        inundations = heightToInundation(onHeights, gauge)
        kernel = tohokuKDE(inundations, offHeights, **kwargs)
        inundationKdes.append(kernel)

    return inundationKdes


#def buildGaugeLikelihoods(gaugeFile=gauges_path, tohokuFile=amp_data_path, heightFile=height_llh_path, inundationFile=inun_llh_path, offshorePoints=500, gaugeIds=-1, **kwargs):
#    """
#
#    :param gaugeFile:
#    :param tohokuFile:
#    :param heightFile:
#    :param inundationFile:
#    :param offshorePoints:
#    :param gaugeIds:
#    :return:
#    """
#
#    print("Loading gauges from " + gaugeFile)
#    gauges = list(np.load(gaugeFile))
#    gauges = [from_json(gauge) for gauge in gauges]
#
#    if gaugeIds != -1:
#        gauges = [gauges[i] for i in gaugeIds]
#
#    print("Making KDEs...")
#    heightKdes, inundationKdes = makeGaugeKDEs(gauges, tohokuFile, **kwargs);
#
#    # These are the values that we will fill in
#    maxOffShoreHeight = max(
#        [kde.dataset[1, :].max() for kde in heightKdes])  # max([ val[:,1].max() for val in vals ]);
#    offShoreHeights = np.linspace(0.0, maxOffShoreHeight, num=offshorePoints)
#
#    heightLikelihood = np.zeros((len(offShoreHeights), len(gauges)))
#    inundationLikelihood = np.zeros((len(offShoreHeights), len(gauges)))
#
#    print("Building gauge likelihoods...")
#    for gid, gauge in enumerate(gauges):
#        print("starting gauge " + str(gid) + "...")
#        if (gauge.kind[1]):
#            ##fix bug that existed in gauge 5 at one point
#            # if (gauge.kind[1] == 'norm' and len(gauge.height_params) == 3):
#            #    print("WARNING: Editing gauge.height_params")
#            #    gauge.height_params = gauge.height_params[-2:]
#            # print(gauge.height_params)
#            heightLikelihood[:, gid], condDist = computeHeightLikelihoodPdf(heightKdes[gid], gauge, offShoreHeights)
#            np.save('condDist_ht_' + str(gid) + '.npy', condDist)
#
#        if (gauge.kind[2]):
#            inundationLikelihood[:, gid], condDist = computeInundationLikelihoodPdf(inundationKdes[gid], gauges[gid],
#                                                                                    offShoreHeights)
#            np.save('condDist_inun_' + str(gid) + '.npy', condDist)
#
#    outputData = np.insert(heightLikelihood, 0, offShoreHeights, axis=1)
#    np.save(heightFile, outputData)
#
#    outputData = np.insert(inundationLikelihood, 0, offShoreHeights, axis=1)
#    np.save(inundationFile, outputData)


# # function to compute the height likelihood
# def computeHeightLikelihoodPdf(kde, gauge, offShoreHeights):
#     """
# 
#     :param kde:
#     :param gauge:
#     :param offShoreHeights:
#     :return:
#     """
#     maxOnShoreHeight = kde.dataset[0, :].max()
#     gaugePdf = gauge.height_dist.pdf
#     if gauge.kind[1] == "chi2" and gauge.height_params[0] < 2.0:  # chi2 can have infinite pdf
#         print("chi2: x is from ",gauge.height_params[1],"to",maxOnShoreHeight)
#         x = np.linspace(gauge.height_params[1] + 0.001, maxOnShoreHeight, num=1000)
#         wt = trapRuleWeights(x);  # trapezoidal rule
# 
#     # #Gauss-Hermite quadrature for normal distributions didn't work - when there is very little overlap
#     # #between a Gauge distribution and the Tohoku KDE, computing the conditional distribution rapidly 
#     # #became numerically unstable. It seems that we need an integration rule that covers the whole 
#     # #region of possible values.
#     # elif gauge.kind[1] == "norm":  
#     #     # use optimal quadrature for normal distributions
#     #     # here we build the pdf into the quadrature rule so we set the "pdf" to the constant function
#     #     x,wt = gaussHermite(100, mu=gauge.height_params[0], sig=gauge.height_params[1]);
#     #     gaugePdf = lambda x:1
# 
#     else:
#         x = np.linspace(0.0, maxOnShoreHeight, num=1000)
#         wt = trapRuleWeights(x);  # trapezoidal rule
# 
#     return computeLikelihoodPdf(kde.pdf, gaugePdf, x, wt, offShoreHeights)
# 
# 
# # function to compute the inundation likelihood
# def computeInundationLikelihoodPdf(kde, gauge, offShoreHeights):
#     """
#     quadrature rule (long-term it would be better to use a gaussian quadrature
#      rule with points/weights picked by the gauge distribution)
#     :param kde:
#     :param gauge:
#     :param offShoreHeights:
#     :return:
#     """
#     maxInundation = kde.dataset[0, :].max()
#     gaugePdf = gauge.inundation_dist.pdf
#     if gauge.kind[2] == "chi2" and gauge.inundation_params[0] < 2.0:  # chi2 can have infinite pdf
#         x = np.linspace(gauge.inundation_params[1] + 0.001, maxInundation, num=1000)
#         wt = trapRuleWeights(x)  # trapezoidal rule
# 
#     #removed Gauss-Hermite quadrature for normal distributions per comment in computeHieghtLikelihoodPdf
#     #elif gauge.kind[2] == "norm":  
#     #    # use optimal quadrature for normal distributions
#     #    # here we build the pdf into the quadrature rule so we set the "pdf" to the constant function
#     #    x,wt = gaussHermite(100, mu=gauge.inundation_params[0], sig=gauge.inundation_params[1]);
#     #    gaugePdf = lambda x:1
# 
#     else:
#         x = np.linspace(0.0, maxInundation, num=1000)
#         wt = trapRuleWeights(x)  # trapezoidal rule
# 
#     return computeLikelihoodPdf(kde.pdf, gaugePdf, x, wt, offShoreHeights)
# 
# 
# def computeLikelihoodPdf(kdePdf, gaugePdf, x, wt, offShoreHeights):
#     """
# 
#     :param kdePdf:
#     :param gaugePdf:
#     :param x:
#     :param wt:
#     :param offShoreHeights:
#     :return:
#     """
#     gPdf = gaugePdf(x)
#     # test quadrature rule:
#     intGaugePdf = sum(wt * gPdf)
#     #print("intGaugePdf = ",intGaugePdf)
#     if np.abs(intGaugePdf - 1.0) > 0.05:
#         print("WARNING: Integration rule may not be accurate enough. Integration of gauge PDF yielded: {:.6f}".format(
#             intGaugePdf))
# 
#     xy = np.zeros((2, len(x)))
#     xy[0, :] = x
#     likelihood = np.zeros(len(offShoreHeights))
#     condDist = np.zeros((len(offShoreHeights) + 1, len(x) + 1))
#     condDist[0, 1:] = x
#     condDist[1:, 0] = offShoreHeights
#     for yidx, y in enumerate(offShoreHeights):
#         xy[1, :] = y
#         kdeXY = kdePdf(xy)
#         #print("integral across row is",sum(wt*kdeXY))
#         nrm = sum(wt * kdeXY)     # normalization factor
#         if nrm > 0.:
#           kdeXY /= sum(wt * kdeXY)  # conditional distribution
#         else:
#           print("Warning: Conditional distribution is zero for off-shore height =",y)
#         likelihood[yidx] = sum(wt * kdeXY * gPdf)
#         condDist[yidx + 1, 1:] = kdeXY
# 
#     # NOTE: It would probably be faster to fully vectorize the above with something like
#     # xx,yy = np.meshgrid(x,y)
#     # xy=np.vstack([xx.ravel(), yy.ravel()])
#     # likelihood = {matrix multiply involving kde.pdf(xy)}
#     # would need to figure out how to map the weights and gauge pdf values to the meshed values though
#     # And we only need to compute this (roughly) once so I'm skipping this for now.
# 
#     intLikelihood = sum(likelihood * trapRuleWeights(offShoreHeights))
# 
#     # normalize
#     likelihood /= intLikelihood
# 
#     return likelihood, condDist
#
#
# # makeGaugeKDEs() builds a height and inundation KDE for
# # each gauge
# def makeGaugeKDEs(gauges, tohokuFile='amplification_data.npy', **kwargs):
#     """
# 
#     :param gauges:
#     :param tohokuFile:
#     :return:
#     """
#     heightKdes = []
#     inundationKdes = []
# 
#     amplification_data = np.load(tohokuFile)
# 
#     for gid, gauge in enumerate(gauges):
#         print("starting gauge " + str(gid) + "...")
#         # figure out which kde to use for this gauge
#         kdeDistances = np.arange(amplification_data.shape[1]) / 4
#         d_idx = np.argmin(np.abs(kdeDistances - gauge.distance)) + 1
#         print("using amplification data column " + str(d_idx))
# 
#         # height kdes
#         onHeights = amplification_data[:, 0]
#         offHeights = amplification_data[:, d_idx]
#         kernel = tohokuKDE(onHeights, offHeights, **kwargs)
#         heightKdes.append(kernel)
# 
#         # inundation kdes
#         inundations = heightToInundation(onHeights, gauge)
#         kernel = tohokuKDE(inundations, offHeights)
#         inundationKdes.append(kernel)
# 
#     return heightKdes, inundationKdes


def heightToInundation(onHeights,gauge):
    """

    :param onHeights:
    :param gauge:
    :return:
    """
    #return np.power(onHeights,4/3) * 0.06 * np.cos(gauge.beta) / (gauge.n**2)
    return np.power(np.maximum(onHeights,0),4/3) * 0.06 * np.cos(np.pi*gauge.beta/180.0) / (gauge.n**2)


#def plotGaugeLikelihoods(inputFile=height_llh_path,outputFile=gauge_output):
#    """
#
#    :param inputFile:
#    :param outputFile:
#    :return:
#    """
#    heightLikelihood = np.load(inputFile)
#    offShoreHeights = heightLikelihood[:,0]
#    heightLikelihood = heightLikelihood[:,1:]
#    fig, ax = plt.subplots()
#    for gid in range(heightLikelihood.shape[1]):
#        ax.plot(offShoreHeights,heightLikelihood[:,gid],label="Gauge "+str(gid))
#    #ax.set_xlim([0.0,max(offShoreHeights/2)])
#    plt.xlabel("Offshore wave height (Geoclaw output)")
#    plt.ylabel("Likelihood")
#    plt.legend()
#    plt.savefig(outputFile)
#    plt.close()


def plotGaugeLikelihoods(inputFile,outputFile=None,xlabel="Offshore wave height (Geoclaw output)",xlim=None):
    """ 

    :param inputFile:
    :param outputFile:
    :return:
    """
    heightLikelihood = np.load(inputFile)
    offShoreHeights = heightLikelihood[:,0]
    heightLikelihood = heightLikelihood[:,1:]

    fig, ax = plt.subplots()
    for gid in range(heightLikelihood.shape[1]):
        ax.plot(offShoreHeights,heightLikelihood[:,gid],label="Gauge "+str(gid))
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel("Likelihood")
    plt.legend()

    #print to file if one is specified
    if outputFile is not None:
        plt.savefig(outputFile)
#        plt.close()
        print("Wrote:",outputFile)


def plotGaugeCompare(lhFile,gauges,gtype=1,outputFolder=None,xlabel="Offshore wave height (Geoclaw output)",xlim=None):
    """ 
    Plot a comparison between the gauge PDF (from Wichmann) and the likelihood
    :param lhFile: Holds likelihood associated with the gauges
    :param gauges: List of gauges
    :param gtype:  Type of gauge pdf to use (1 for height, 2 for inundation)
    :param outputFolder: Save figures in this folder (if None, figures not saved to file)
    :return:
    """
    heightLikelihood = np.load(lhFile)
    offShoreHeights = heightLikelihood[:,0]
    heightLikelihood = heightLikelihood[:,1:]
    
    #IDs of gauges that have this kind of observation (e.g., height or inundation)
    gids = [i for i, gauge in enumerate(gauges) if gauge.kind[gtype] is not None]

    for i,gid in enumerate(gids):
        fig, ax = plt.subplots()
        
        if gtype == 1:
            gpdf = gauges[gid].height_dist.pdf
        if gtype == 2:
            gpdf = gauges[gid].inundation_dist.pdf

        ax.plot(offShoreHeights,heightLikelihood[:,i], label="Gauge "+str(gid)+" (Likelihood)")
        ax.plot(offShoreHeights,gpdf(offShoreHeights), label="Gauge "+str(gid)+" (Wichmann)")
        
        if xlim is not None:
            ax.set_xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel("Likelihood")
        plt.legend()

        #print to file if output folder is specified
        if outputFolder is not None:
            if gtype == 1:
                typeStr = "ht"
            if gtype == 2:
                typeStr = "inun"
            outputFile = outputFolder+"/gauge_compare_"+typeStr+"_"+str(gid)+".png"
            plt.savefig(outputFile)
            #plt.close()
            print("Wrote:",outputFile)


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

#buildGaugeLikelihoods()
