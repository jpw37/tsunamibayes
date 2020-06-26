import numpy as np

def abrahamson(mag, dist, V_S30):
    """Computes the spectral acceleration model of Abrahamson, et al
    for an interface event.

    To compute PGA_1000, run the model with VS30 = 1000, PGA = 0 and associated
    regression parameters (those listed for period = 0 hz.)

    Parameters
    ----------
    mag : float
        Earthquake moment magnitude, passed into the run_model function as 'M'
    dist : float
        Distance to fault (km), passed into the run_model function as 'R'
    V_S30 : float
        Shear wave velocity in the top 30 meters (m/s), passed into run_model as VS30

    Returns
    ----------
    log : float
        Spectral acceleration in units of g
    np.sqrt(phi**2 + tau**2) : float
        Standard deviation of ln(PSA)- the natural log of the peak spectral acceleartion in units of g
        Given by the root sum squared of the intraevent's standard deviations 
        (The variability among observations within an event).

    """
    def run_model(M, R, PGA, VS30):
        """Helps determine the value of PGA_100 and computes the spectral acceleration 
        model of Abrahamson for period specified period.

        Parameters
        ----------
        M : float
            Moment magnitude of earthquake
        R : float
            Distance to fault (km)
        V_lin : float
            Regression parameter for velocity
        b : float
            Regression parameter for site response scaling
        thetas : (list or ndarray)
            Period dependent regression parameters
        C1 : float
            Regression parameter
        deltaC : float
            Difference between C1 for slab and interface
        PGA : float
            Median peak ground acceleration for VS30 = 1000
        VS30 : float
            Shear wave velocity in the top 30 meters (m/s)
        backarc : bool
            True for backarc site
        tau : float
            Interevent standard deviation

        Returns
        -------
        log : float
            Spectral acceleration in units of g.
            
        """
        # Period-independent coefficients
        n = 1.18
        c = 1.88
        C4 = 10.0
        backarc = False

        # coefficients for calulating PGA
        if PGA == 0:
            V_lin = 865.1
            b = -1.186
            
            a = [2.340, -1.044, 0.1, 0.59, 0, -0.00705, 0, 0, .4,
            1.73, 0.017, 0.818, -0.0135, -0.223, 0, 0]
            
            C1 = 8.2
            deltaC = -1.
            PGA_adjust = 1.044

        # coefficients for period = 1hz
        else:
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
    PGA_1000 = np.exp(run_model(mag, dist, 0, 1000.))
    
    # Run model for period = 1hz
    log = run_model(mag, dist, PGA_1000, V_S30)

    phi = 0.60 # intraevent standard deviation
    tau = 0.43 # interevent standard deviation

    return log, np.sqrt(phi**2 + tau**2)
