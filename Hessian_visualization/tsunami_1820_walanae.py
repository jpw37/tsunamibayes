"""This script loads the MCMC data for the tsunami event in 1852 and set some useful
variables.
"""


from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).absolute().parent
DATA_DIR = WORK_DIR / "data/1820_Walanae"


# Load the MCMC data
# List the folder corresponding to each chain
chain_folder = sorted([Path(dd) for dd in glob(str(DATA_DIR / "*")) if Path(dd).is_dir()])
# Load the MCMC results
mcmc_raw_data = {}  # Instantiate a dictionary to store the MCMC data
# These items are the descriptions of the files we want to use in each chain folder.
# These are also the name of the files.
file_desc = ["bayes_data", "model_output", "model_params", "samples"]

for ii, ff in enumerate(chain_folder):
    # Instantiate subdictionary. The identifier is the path to the chain folder.
    mcmc_raw_data.update({f"chain{ii}": {"identifier": ff}})

    # Load each mcmc file and store inside the dictionary
    for desc in file_desc:
        df = df = pd.read_csv(ff / f"{desc}.csv", index_col=0)
        mcmc_raw_data[f"chain{ii}"].update({desc: df})
# Burnin andd autocorrelation
burnin = 3000
thin = 100
# Combine the data across different chains
mcmc_data = {}
for desc in file_desc:
    # Use the data in chain 0 to initialize a data frame to store the combined values.
    data = mcmc_raw_data["chain0"][desc].iloc[burnin::thin]
    for key in list(mcmc_raw_data)[1:]:
        new_rows = mcmc_raw_data[key][desc].iloc[burnin::thin]
        data = pd.concat([new_rows, data.loc[:]]).reset_index(drop=True)
    mcmc_data.update({desc: data})


# The evaluation point $x_0$ and $Q_0$ that are used to estimate the derivative needs to
# be directly related, i.e., $Q_0$ is the output of $x_0$ through the model mapping (or
# posterior mapping if we are using the posterior). Taking the mean of the input and
# output samples doesn't guarantee that $x_0$ is related to $Q_0$, especially if the
# mapping is non-linear. What we will do, instead, is finding the sample $x$ that is the
# closest to the evaluation parameter point $x_0$, and set such $x$ as $x_0$. Then, we
# take the corresponding output sample as $Q_0$. The notion of distance of sample $x$ to
# the mean will be measured through the Euclidean distance between the sample and the
# mean, after we scale $x$ to reconcile the difference in the physical units.

# Some evaluation points
# These indices will be used to get $x_0$ and $Q_0$
# MAP
logposterior_samples = mcmc_data["bayes_data"].loc[:, "posterior_logpdf"]
idx_map = np.argmax(logposterior_samples)
# MLE
loglikelihood_samples = mcmc_data["bayes_data"].loc[:, "llh"]
idx_mle = np.argmax(loglikelihood_samples)
# Parameter mean  # Rake doesn't vary, so exclude this parameter
model_params = mcmc_data["model_params"].loc[
    :,
    [
        "latitude",
        "longitude",
        "length",
        "width",
        "slip",
        "strike",
        "dip",
        "depth",
        "depth_offset",
    ],
]
model_params_mean = np.mean(model_params, axis=0)
model_params_std = np.std(model_params, axis=0)
model_params_scaled_dist = np.linalg.norm(
    (model_params - model_params_mean) / model_params_std, axis=1
)
idx_mean = np.argmin(model_params_scaled_dist)


# Samples
input_samples = model_params
# As output, we will exclude inundation, since it is related to the height
output_samples = mcmc_data["model_output"].loc[
    :,
    [
        "Bulukumba height",
        "Bulukumba arrival",
        "Sumenep height",
        "Sumenep arrival",
        "Nipa-Nipa height",
        "Bima height",
    ],
]
posterior_samples = np.exp(logposterior_samples)
likelihood_samples = np.exp(loglikelihood_samples)
