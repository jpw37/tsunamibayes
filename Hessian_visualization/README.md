# Tsunami 1820 and 1852: Hessian estimation from MCMC samples

This repo is related to the project in https://github.com/jpw37/tsunamibayes

In this repo, we use the MCMC results from the historical tsunami events in Indonesia in
1820 and 1852 and estimate the Hessian of the posterior near the MAP point. The Hessian
can be used to analyze the sesitivity of the posterior with respect to the parameters and
to check the resolution of the posterior samples.

The main files are the 3 Jupyter notebooks, each for different event or different source
of earthquake.
1. [tsunami_1852.ipynb](https://github.com/yonatank93/tsunami_project/blob/main/tsunami_1852.ipynb) </br>
   This notebook is for estimating Hessian corresponding to the 1852 Banda Sea tsunami
   event.
2. [tsunami_1820_flores.ipynb](https://github.com/yonatank93/tsunami_project/blob/main/tsunami_1820_flores.ipynb) </br>
   This notebook is for estimating Hessian corresponding to the 1820 Sulawesi tsunami
   event, when we assume that the earthquake came from Flores fault.
3. [tsunami_1820_walanae.ipynb](https://github.com/yonatank93/tsunami_project/blob/main/tsunami_1820_walanae.ipynb) </br>
   This notebook is for estimating Hessian corresponding to the 1820 Sulawesi tsunami
   event, when we assume that the earthquake came from Walanae fault.
   
Other important files include:
* [sensitivity_regression.py](https://github.com/yonatank93/tsunami_project/blob/main/sensitivity_regression.py) </br>
  This file contains the main routine to estimate the Jacobian and Hessian from MCMC
  samples via linear regression.
* [utils.py](https://github.com/yonatank93/tsunami_project/blob/main/utils.py) </br>
  This file contains some utility functions to pre-process the MCMC samples, prior to
  estimating the derivative.
  
Additionally, data, results, and generated figures are included in their respective
folder.
