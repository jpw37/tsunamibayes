"""This module contains functions estimate snsitivity matrix of the model at a set of
parameter value with the help of linear regression. Note that currently only
``np.linalg.lstsq`` solver is implemented.

Included in this module:
* a function to estimate the first order sensitivity values
* a function to estimate the first and second order sensitivity values

author: Yonatan Kurniawan
email: kurniawanyo@outlook.com
"""

import numpy as np


def _design_matrix_first_order(dX, K, M, N):
    """Generate a design matrix that is used in the first order sensitivity estimation by
    regression.

    Parameters
    ----------
    dX: np.ndaray (K, N,)
        An array containing the difference between the sampled parameters and the
        reference.
    K, M, N: int
        Number of samples, model outputs, and input parameters, respectively.

    Returns
    -------
    D: np.ndarray (K*M, M*N,)
        Design matrix for first order sensitivity by regression.
    """
    # Generate the design matrix. The sensitivity matrix will have M rows and N columns,
    # or in other words, we have M * N unknowns. So, the design matrix should have M * N
    # columns. If we stack the outputs across all samples, then the design matrix should
    # have K * M rows.
    D = np.zeros((K * M, M * N))
    # Populate this zero matrix.
    for k, dx in enumerate(dX):
        kidx_start = k * M
        for m in range(M):
            kidx = kidx_start + m
            midx_start = m * N
            midx_end = (m + 1) * N
            D[kidx, midx_start:midx_end] = dx
    return D


def _check_dimensions(x0, Q0, x_samples, Q_samples):
    """Get the dimension of the problem and fix the format of the samples, as needed.

    Parameters
    ----------
    x0: np.ndarray (N,)
        The parameter values on which the sensitivity matrix will be estimated.
    Q0: np.ndarray (M,)
        The output of the model evaluted at ``x0``.
    x_samples: np.ndarraY (K, N,)
        Parameter values sampled around ``x0``. As a rule of thumb, we need
        :math:`K \geq M \times N`.
    Q_samples: np.ndarray (K, M,)
        Output values evaluated at each ``x_samples`` point.

    Returns
    -------
    K, M, N: int
        Number of samples, model outputs, and input parameters, respectively.
    x_samples: np.ndarray (K, N,)
        Parameter values sampled around ``x0``.
    Q_samples: np.ndarray (K, M,)
        Output values evaluated at each ``x_samples`` point.
    """
    # Extract the dimensionality
    try:
        M = len(Q0)
    except TypeError:
        Q0 = np.array([Q0])
        M = 1
    try:
        N = len(x0)
    except TypeError:
        x0 = np.arrau([x0])
        N = 1
    K = len(x_samples)

    # Fix the dimensionality of the samples array, as needed
    if M == 1:
        Q_samples = Q_samples.reshape((K, M))
    if N == 1:
        x_samples = x_samples.reshape((K, N))

    return K, M, N, x_samples, Q_samples


def first_order(x0, Q0, x_samples, Q_samples, solve=True, full_output=False):
    """Use linear regression to estimate the first order sensitivity matrix of a model.

    Parameters
    ----------
    x0: np.ndarray (N,)
        The parameter values on which the sensitivity matrix will be estimated.
    Q0: np.ndarray (M,)
        The output of the model evaluted at ``x0``.
    x_samples: np.ndarraY (K, N,)
        Parameter values sampled around ``x0``. As a rule of thumb, we need
        :math:`K \geq M \times N`.
    Q_samples: np.ndarray (K, M,)
        Output values evaluated at each ``x_samples`` point.
    solve: bool (optional)
        This is a flag to solve the linear regression. If this is set to ``False``, then
        this function will only return vector ``dQ`` and matrix ``D`` that can be used
        in other regression algorithm, such as ridge, lasso, or elastic net regression.
    full_output: bool (optional)
        A flag to return the full output, which also return the vector of model outputs
        differences, the design matrix, and other outputs of ``np.linalg.lstsq``.

    Returns
    -------
    S: np.ndarray(M, N,)
        The estimated first order sensitivity matrix.
    full_output
        dQ: np.ndarray (K * M,)
            Model outputs differences.
        D: np.ndarray (K, M * N)
            Design matrix in the linear least squares regression.
        Output of ``np.linalg.lstsq``.

    Raises
    ------
    ValueError
        When the numbers of samples in ``x_samples`` and ``Q_samples`` are different.
    ValueError
        When the number of samples is less than the number of parameters. In this case,
        the linear regression won't work since we will have an underconstrained problem.
    """
    # Get dimensionality
    K, M, N, x_samples, Q_samples = _check_dimensions(x0, Q0, x_samples, Q_samples)

    # Check dimensionality
    Kmin = int(N)
    if len(Q_samples) != K:
        raise ValueError("The number of sampled parameters and outputs are different")
    if K < Kmin:
        raise ValueError(f"Please provide at least {Kmin} samples")

    # Generate delta vectors for the outputs and parameters
    # Delta vector of the outputs
    dQ = (Q_samples - Q0.reshape((1, M))).flatten()
    dX = x_samples - x0.reshape((1, N))  # Delta matrix of the parameters

    # Generate the design matrix.
    D = _design_matrix_first_order(dX, K, M, N)

    if solve:
        # Perform least squares regression
        lstsq_result = np.linalg.lstsq(D, dQ, rcond=-1)
        # Construct the sensitivity matrix
        S = lstsq_result[0].reshape((M, N))

        if full_output:
            return S, dQ, D, lstsq_result
        else:
            return S
    else:
        return dQ, D


def first_second_order(x0, Q0, x_samples, Q_samples, solve=True, full_output=False):
    """Use linear regression to estimate the first order sensitivity matrix of a model.

    Parameters
    ----------
    x0: np.ndarray (N,)
        The parameter values on which the sensitivity matrix will be estimated.
    Q0: np.ndarray (M,)
        The output of the model evaluted at ``x0``.
    x_samples: np.ndarraY (K, N,)
        Parameter values sampled around ``x0``. As a rule of thumb, we need
        :math:`K \geq M \times N`.
    Q_samples: np.ndarray (K, M,)
        Output values evaluated at each ``x_samples`` point.
    solve: bool (optional)
        This is a flag to solve the linear regression. If this is set to ``False``, then
        this function will only return vector ``dQ`` and matrix ``D`` that can be used
        in other regression algorithm, such as ridge, lasso, or elastic net regression.
    full_output: bool (optional)
        A flag to return the full output, which also return the vector of model outputs
        differences, the design matrix, and other outputs of ``np.linalg.lstsq``.

    Returns
    -------
    S1: np.ndarray(M, N,)
        The estimated first order sensitivity matrix.
    S2: np.ndarray(M, N, N,)
        The estimated second order sensitivity tensor.
    full_output
        dQ: np.ndarray (K * M,)
            Model outputs differences.
        D: np.ndarray (K, M * N)
            Design matrix in the linear least squares regression.
        Output of ``np.linalg.lstsq``.

    Raises
    ------
    ValueError
        When the numbers of samples in ``x_samples`` and ``Q_samples`` are different.
    ValueError
        When the number of samples is less than the suggested minimum number of samples,
        which is :math:`(N^2 + 3 N) / 2`. In this case, the linear regression won't work
        since we will have an underconstrained problem.
    """
    # Get dimensionality
    K, M, N, x_samples, Q_samples = _check_dimensions(x0, Q0, x_samples, Q_samples)

    # Check dimensionality
    Kmin = int((N**2 + 3 * N) / 2)
    if len(Q_samples) != K:
        raise ValueError("The number of sampled parameters and outputs are different")
    if K < Kmin:
        raise ValueError(f"Please provide at least {Kmin} samples")

    # Generate delta vectors for the outputs and parameters
    # Delta vector of the outputs
    dQ = (Q_samples - Q0.reshape((1, M))).flatten()
    dX = x_samples - x0.reshape((1, N))  # Delta matrix of the parameters

    # Generate a design matrix. This matrix is typically huge.
    D = np.zeros((K * M, M * Kmin))
    # For the first M * N columns of this matrix, it will be the same as the design
    # matrix used for the first order estimation.
    D[:, : (M * N)] = _design_matrix_first_order(dX, K, M, N)
    # Populate the rest of the columns. There is a pattern, where each block is something
    # like (dx1**2, dx2**2, ..., dx1*dx2, dx1*dx3, ...).
    D2 = D[:, (M * N) :]  # This is just to make indexing easier
    # Length of each block of the acceleration term
    nacc_term = int(D2.shape[1] / M)
    for k, dx in enumerate(dX):
        kidx_start = k * M
        # Generate blocks
        first_N_col = dx**2 / 2
        the_rest = []
        for ii in range(N):
            for jj in range(ii + 1, N):
                the_rest = np.append(the_rest, dx[ii] * dx[jj])
        acc_block = np.append(first_N_col, the_rest)
        for m in range(M):
            kidx = kidx_start + m
            midx_start = m * nacc_term
            midx_end = (m + 1) * nacc_term
            D2[kidx, midx_start:midx_end] = acc_block

    if solve:
        # Perform least squares regression
        lstsq_result = np.linalg.lstsq(D, dQ, rcond=-1)

        # Construct the sensitivity matrix
        Svec = lstsq_result[0]
        # First order sensitivity matrix
        S1 = Svec[: (M * N)].reshape((M, N))
        # Second order sensitivity matrix
        # Get the values and reshape. Each row contains values for each output
        Svec_2 = Svec[(M * N) :].reshape((M, -1))
        S2 = np.empty((M, N, N))  # Instantiate the sensitivity tensor
        for m, vals in enumerate(Svec_2):
            # val contains all values needed to construct second order sensitivity matrix
            # for each output.
            diag = vals[:N]  # Diagonal elements of the sensitivity matrix
            # Instantiate the sensitivity matrix and populate the diagonal elements
            A = np.diag(diag)  # The size of this matrix is N by N
            # Get the off-diagonal elements
            off_diag = vals[N:]
            # Instantiate a zero matrix to populate the off-diagonal elements
            tri_mat = np.zeros_like(A)
            for row in range(N - 1):
                for col in range(row + 1, N):
                    tri_mat[row, col] = off_diag[0]
                    off_diag = np.delete(off_diag, 0)
            # Add the off-diagonal elements to the sensitivity matrix
            A += tri_mat  # Upper diagonal elements
            A += tri_mat.T  # Lower diagonal elements
            # Store this completed sensitiity matrix to S2 tensor
            S2[m] = A

        if full_output:
            return S1, S2, dQ, D, lstsq_result
        else:
            return S1, S2
    else:
        return dQ, D


if __name__ == "__main__":
    # Test the functions to estimate the sensitivity matrix via linear regression using
    # some linear model.
    np.random.seed(2023)

    # Dimensionality
    M = 5
    N = 3
    K = 20

    tlist = np.logspace(-2, 1, M)

    def generate_design_matrix(tlist):
        D = np.array([tlist**n for n in range(N)]).T
        return D

    def linear_model(x):
        """Test linear model with 3 parameters."""
        D = generate_design_matrix(tlist)
        return D @ x

    # Evaluation parameters and output
    x0 = np.random.randn(N)
    Q0 = linear_model(x0)

    # Generate samples
    xsamples = np.random.normal(loc=x0, size=(K, N))
    Qsamples = np.array([linear_model(x) for x in xsamples])

    # Estimate the first and second order sensitivity matrices using the function above.
    S1_est, S2_est = first_second_order(x0, Q0, xsamples, Qsamples)

    # For a linear model, we know that the first order sensitivity matrix would just be
    # the design matrix. So, we will be using this fact to test our function.
    S1_ref = generate_design_matrix(tlist)
    assert np.allclose(S1_est, S1_ref)

    # Also for a linear model, we know that the second order sensitivity (which is the
    # same as the acceleration matrix) should be zero.
    S2_ref = np.zeros((M, N, N))
    assert np.allclose(S2_est, S2_ref)
