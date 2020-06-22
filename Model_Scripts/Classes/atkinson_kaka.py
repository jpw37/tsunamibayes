import numpy as np

def convert_to_MMI(log_SA, M, D):
    """Uses the Atkinson and Kaka model to convert PSA (1 Hz) to MMI.

    Parameters
    ----------
    log_SA : float
        logarithm of the spectral acceleration
    M : float
        magnitude of the earthquake on the Ritcher Scale    
    D : float
        The shortest distance between the observation site and earthquake epicenter (km) as computed in distance.py
    
    Returns
    -------
    MMI : float
        The associated value on the Modified Mercalli intensity scale
    sigma : float
        The root sum square of the standard deviation?
    """

    C = [3.23, 1.18, 0.57, 2.95, 1.92, -0.39, 0.04]
    log_YI5 = 1.50

    sigma_IMMI = 0.84
    sigma_MMI = 0.73
    sigma = np.sqrt(sigma_IMMI**2 + sigma_MMI**2)

    if log_SA <= log_YI5:
        MMI = C[0] + C[1]*log_SA + C[4] + C[5]*M + C[6]*np.log(D)
    else:
        MMI = C[2] + C[3]*log_SA + C[4] + C[5]*M + C[6]*np.log(D)

    return MMI, sigma
