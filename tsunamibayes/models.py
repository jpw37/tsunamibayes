import numpy as np

def inundation(wave_height,beta,n):
    val = wave_height**(4/3)*0.06*np.cos(np.deg2rad(beta))/(n**2)
    return max(val,0)
