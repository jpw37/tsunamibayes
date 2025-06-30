import numpy as np


def normalize_samples(samples, shift=0.0, scale=1.0):
    """This function normalizes the samples using the mean and standard deviation.
    The normalization is done by:
    
    .. math::
       normalized_samples = (samples - shift) / scale

    Parameters
    ----------
    samples: pd.DataFrame or np.ndarray
        Original samples to normalize.
    shift, scale: float or np.ndarray (optional)
        Shifting and scaling values.

    Returns
    -------
    np.ndarray
        Normalized samples.
    """
    normalized_samples = (samples - shift) / scale
    return normalized_samples


def get_samples_subset(samples, center=0.0, r=1.0):
    """This function extract a subset of the samples. The subset is obtained by
    drawing a hypersphere centered at a point and only retrieve points inside the
    hypersphere.

    Parameters
    ----------
    samples: np.ndarray
        Samples that we want to filter.
    center: float or np.ndarray
        Center of the hypersphere.
    r: float
        Radius of the hypersphere.

    Returns
    -------
    np.ndarray
        Filtered samples.
    np.ndarray
        Index to retrieve the subset of the samples.
    """
    # Compute the distance to the center
    dist = np.linalg.norm(samples - center, axis=1)
    # Get index of samples that lie within the hypersphere
    idx = np.where(dist < r)[0]

    return samples[idx], idx