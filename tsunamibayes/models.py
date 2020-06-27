import numpy as np

def inundation(wave_height,beta,n):
    """Computes the inundation distance of the tsunami wave based on wave height.
    
    Parameters
    ----------
    wave_height : float
        The wave height at the shore (in meters).
    beta : float
        The gauge's beta value, the uniform slope of the shoreline (degrees). 
        (FIXME: What does beta stand for in banda_1852 gauges?).
    n : float
        The Manning's coefficient for the surface roughness of the shoreline. 
    
    Returns
    -------
    val : float
        The inundation/runup distance (in meters).
        If the calculation produces a negative value, function returns 0.
    """
    val = wave_height**(4/3)*0.06*np.cos(np.deg2rad(beta))/(n**2)
    return max(val,0)

def abrahamson(type,mag,dist,V_S30,backarc=False):
    """Computes PGA_1000 by running the model with VS30 = 1000, PGA = 0 and associated
    regression parameters (those listed for period = 0 hz.), then uses the computed value
    to computes the spectral acceleration model of Abrahamson, et al for the specific interface event.

    Parameters
    ----------
    type : string
        The type of the model desired. Generally can be either 'PGA' to calculate
        the general Peak Ground acceleartion or '1HZ' to run the model when the earthquake's
        period is 1 HZ.
    mag : float
        Moment magnitude
    dist : float
        Distance to fault (km). Passed into run_model function as 'R'.
    V_S30 : float
        Shear wave velocity in the top 30 meters (m/s).
        Passed into run_model function as 'VS30'.
    backarc : bool
        True if the location of the model is on the backarc side of fault.
        False for modeling in the forarc side of a fault. Default is False. 

    Returns
    ----------
    spec_accel : float
        Spectral acceleration in units of g. (m/s^2)
    sigma : float
        Standard deviation of ln(PSA) in units of g. (m/s^2)
        The root sum squars of the interevent's standard deviations.
        
    """

    def run_model(type,M,R,PGA_1000,VS30,backarc):
        """Computes the spectral acceleration model of Abrahamson, et al
        for an interface event.
        
        Parameters
        ----------
        type : string
            The type of the model desired. Generally can be either 'PGA' to calculate
            the general Peak Ground acceleartion or '1HZ' to run the model when the earthquake's
            period is 1 HZ.
        M : float
            The earthquake's moment magnitude.
        R : float
            Distance to fault (km).
        PGA_1000 : float
            The peak ground acceleration (m/s^2) when the shear wave velocity in the top 
            30 meters (V_S30) is 1000 m/s. 
        VS30 : float
            Shear wave velocity in the top 30 meters (m/s).
        backarc : bool
            True if the location of the model is on the backarc side of fault.
            False for modeling in the forarc side of a fault. 
        
        Returns
        -------
        log : float
            The calculated spectral acceleration in units of g. (m/s^2)
        """

        # Period-independent coefficients
        n = 1.18
        c = 1.88
        C4 = 10.0

        # coefficients for calulating PGA
        if type == 'PGA':
            V_lin = 865.1
            b = -1.186

            a = [2.340, -1.044, 0.1, 0.59, 0, -0.00705, 0, 0, .4,
            1.73, 0.017, 0.818, -0.0135, -0.223, 0, 0]

            C1 = 8.2
            deltaC = -1.
            PGA_adjust = 1.044

        # coefficients for period = 1hz
        elif type == '1HZ':
            V_lin = 400.0
            b = -1.955
            C1 = 8.1
            deltaC = -0.9
            PGA_adjust = 0

            # Abrahamson's spreadsheet coefficients
            a = [1.851, -.698, .1, .68, 0, -.00645, 0, 0,
                 .4, 1.73, .01, 1.402, -.0363, -.261]

        # Magnitude scaling
        if M > C1:
            fmag = a[4]*(M-C1)+a[12]*(10-M)**2
        else:
            fmag = a[3]*(M-C1) + a[12]*(10-M)**2


        # Site response scaling
        if VS30 > 1000: # Cap VS30 at 1000 m/s
            Vstar = 1000.
        else:
            Vstar = VS30

        if VS30 < V_lin:
            fsite = a[11]*np.log(Vstar/V_lin)-b*np.log(PGA+c) + \
                    b*np.log(PGA + c*(Vstar/V_lin)**n)
        else:
            fsite = (a[11]+ b*n)*np.log(Vstar/V_lin)

        # Forearc vs Backarc
        if backarc:
            fFABA = a[14]+a[15]*np.log(max(R,100)/40)
        else:
            fFABA = 0

        # Full equation:
#         log = a[0] + a[3]*deltaC + (a[1] + a[2]*(M-7.8))* \
#               np.log(R+C4*np.exp(a[8]*(M-6))) + \
#               a[5]*R + fmag + fFABA + fsite

        log = a[0] + (a[1] + a[2]*(M-7.8))* \
              np.log(R+C4*np.exp(a[8]*(M-6))) + \
              a[5]*R + fmag + fFABA + fsite + PGA_adjust

        return log


    # Calculate PGA 1000
    PGA_1000 = np.exp(run_model('PGA',mag,dist,0,1000.))

    # Run model for period = 1hz
    log = run_model(type, mag, dist, PGA_1000, V_S30)

    phi = 0.60 # intraevent standard deviation
    tau = 0.43 # interevent standard deviation

    return log, np.sqrt(phi**2 + tau**2)
