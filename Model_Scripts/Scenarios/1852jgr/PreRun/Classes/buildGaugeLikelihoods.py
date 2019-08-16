import matplotlib
matplotlib.use('agg', warn=False, force=True)
from matplotlib import pyplot as plt

import numpy as np
from Gauge import from_json
from tohoku import tohokuKDE

gauges_path = "InputData/gauges.npy"
amp_data_path = "../InputData/amplification_data.npy"
height_llh_path = "../InputData/gaugeHeightLikelihood.npy"
inun_llh_path = "../InputData/gaugeInundationLikelihood.npy"
gauge_output = 'InputData/gauge_llh_ht.png'


def buildGaugeLikelihoods(gaugeFile=gauges_path, tohokuFile=amp_data_path, heightFile=height_llh_path,
                          inundationFile=inun_llh_path, offshorePoints=500, gaugeIds=-1):
    """

    :param gaugeFile:
    :param tohokuFile:
    :param heightFile:
    :param inundationFile:
    :param offshorePoints:
    :param gaugeIds:
    :return:
    """

    print("Loading gauges from " + gaugeFile)
    gauges = list(np.load(gaugeFile))
    gauges = [from_json(gauge) for gauge in gauges]

    if gaugeIds != -1:
        gauges = [gauges[i] for i in gaugeIds]

    heightKdes, inundationKdes = makeGaugeKDEs(gauges, tohokuFile);

    # These are the values that we will fill in
    maxOffShoreHeight = max(
        [kde.dataset[1, :].max() for kde in heightKdes])  # max([ val[:,1].max() for val in vals ]);
    offShoreHeights = np.linspace(0.0, maxOffShoreHeight, num=offshorePoints)

    heightLikelihood = np.zeros((len(offShoreHeights), len(gauges)))
    inundationLikelihood = np.zeros((len(offShoreHeights), len(gauges)))

    for gid, gauge in enumerate(gauges):
        print("starting gauge " + str(gid) + "...")
        if (gauge.kind[1]):
            ##fix bug that existed in gauge 5 at one point
            # if (gauge.kind[1] == 'norm' and len(gauge.height_params) == 3):
            #    print("WARNING: Editing gauge.height_params")
            #    gauge.height_params = gauge.height_params[-2:]
            # print(gauge.height_params)
            heightLikelihood[:, gid], condDist = computeHeightLikelihoodPdf(heightKdes[gid], gauge, offShoreHeights)
            np.save('condDist_ht_' + str(gid) + '.npy', condDist)

        if (gauge.kind[2]):
            inundationLikelihood[:, gid], condDist = computeInundationLikelihoodPdf(inundationKdes[gid], gauges[gid],
                                                                                    offShoreHeights)
            np.save('condDist_inun_' + str(gid) + '.npy', condDist)

    outputData = np.insert(heightLikelihood, 0, offShoreHeights, axis=1)
    np.save(heightFile, outputData)

    outputData = np.insert(inundationLikelihood, 0, offShoreHeights, axis=1)
    np.save(inundationFile, outputData)


# function to compute the height likelihood
def computeHeightLikelihoodPdf(kde, gauge, offShoreHeights):
    """

    :param kde:
    :param gauge:
    :param offShoreHeights:
    :return:
    """
    # quadrature rule (long-term it would be better to use a gaussian quadrature
    # rule with points/weights picked by the gauge distribution)
    maxOnShoreHeight = kde.dataset[0, :].max()
    if gauge.kind[1] == "chi2" and gauge.height_params[0] < 2.0:  # chi2 can have infinite pdf
        x = np.linspace(gauge.height_params[1] + 0.01, maxOnShoreHeight, num=5000)
    else:
        x = np.linspace(0.0, maxOnShoreHeight, num=1000)
    wt = trapRuleWeights(x);  # trapezoidal rule

    return computeLikelihoodPdf(kde.pdf, gauge.height_dist.pdf, x, wt, offShoreHeights)


# function to compute the inundation likelihood
def computeInundationLikelihoodPdf(kde, gauge, offShoreHeights):
    """
    quadrature rule (long-term it would be better to use a gaussian quadrature
     rule with points/weights picked by the gauge distribution)
    :param kde:
    :param gauge:
    :param offShoreHeights:
    :return:
    """
    maxInundation = kde.dataset[0, :].max()
    if gauge.kind[2] == "chi2" and gauge.height_params[0] < 2.0:  # chi2 can have infinite pdf
        x = np.linspace(gauge.inundation_params[1] + 0.01, maxInundation, num=5000)
    else:
        x = np.linspace(0.0, maxInundation, num=1000)
    wt = trapRuleWeights(x)  # trapezoidal rule

    return computeLikelihoodPdf(kde.pdf, gauge.inundation_dist.pdf, x, wt, offShoreHeights)


def computeLikelihoodPdf(kdePdf, gaugePdf, x, wt, offShoreHeights):
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
    if np.abs(intGaugePdf - 1.0) > 0.05:
        print("WARNING: Integration rule may not be accurate enough. Integration of gauge PDF yielded: {:.6f}".format(
            intGaugePdf))

    xy = np.zeros((2, len(x)))
    xy[0, :] = x
    likelihood = np.zeros(len(offShoreHeights))
    condDist = np.zeros((len(offShoreHeights) + 1, len(x) + 1))
    condDist[0, 1:] = x
    condDist[1:, 0] = offShoreHeights
    for yidx, y in enumerate(offShoreHeights):
        xy[1, :] = y
        kdeXY = kdePdf(xy)
        kdeXY /= sum(wt * kdeXY)  # conditional distribution
        likelihood[yidx] = sum(wt * kdeXY * gPdf)
        condDist[yidx + 1, 1:] = kdeXY

    # NOTE: It would probably be faster to fully vectorize the above with something like
    # xx,yy = np.meshgrid(x,y)
    # xy=np.vstack([xx.ravel(), yy.ravel()])
    # likelihood = {matrix multiply involving kde.pdf(xy)}
    # would need to figure out how to map the weights and gauge pdf values to the meshed values though
    # And we only need to compute this (roughly) once so I'm skipping this for now.

    intLikelihood = sum(likelihood * trapRuleWeights(offShoreHeights))

    # normalize
    likelihood /= intLikelihood

    return likelihood, condDist


# makeGaugeKDEs() builds a height and inundation KDE for
# each gauge
def makeGaugeKDEs(gauges, flNm='amplification_data.npy'):
    """

    :param gauges:
    :param flNm:
    :return:
    """
    heightKdes = []
    inundationKdes = []

    amplification_data = np.load(flNm)

    for gid, gauge in enumerate(gauges):
        print("starting gauge " + str(gid) + "...")
        # figure out which kde to use for this gauge
        kdeDistances = np.arange(amplification_data.shape[1]) / 4
        d_idx = np.argmin(np.abs(kdeDistances - gauge.distance)) + 1
        print("using amplification data column " + str(d_idx))

        # height kdes
        onHeights = amplification_data[:, 0]
        offHeights = amplification_data[:, d_idx]
        kernel = tohokuKDE(onHeights, offHeights)
        heightKdes.append(kernel)

        # inundation kdes
        inundations = heightToInundation(onHeights, gauge)
        kernel = tohokuKDE(inundations, offHeights)
        inundationKdes.append(kernel)

    return heightKdes, inundationKdes


def heightToInundation(onHeights,gauge):
    """

    :param onHeights:
    :param gauge:
    :return:
    """
    #return np.power(onHeights,4/3) * 0.06 * np.cos(gauge.beta) / (gauge.n**2)
    return np.power(np.maximum(onHeights,0),4/3) * 0.06 * np.cos(np.pi*gauge.beta/180.0) / (gauge.n**2)


def plotGaugeLikelihoods(inputFile=height_llh_path,outputFile='gauge_llh_ht.png'):
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
    #ax.set_xlim([0.0,max(offShoreHeights/2)])
    plt.xlabel("Offshore wave height (Geoclaw output)")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.savefig(outputFile)
    plt.close()


def trapRuleWeights(x):
    """

    :param x:
    :return:
    """
    wt = np.zeros(len(x))
    wt[1:]  += 0.5*(x[1:]-x[:-1])
    wt[:-1] += 0.5*(x[1:]-x[:-1])
    return wt


#buildGaugeLikelihoods()
