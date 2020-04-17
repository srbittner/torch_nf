""" Mean field functions for V1 analyses in PyTorch."""
import numpy as np
DTYPE = np.float32

def IntegralPhi(mu, delta, num_pts=50):
    """Calculates gaussian integral of :math:`\\phi(z)` given mu and delta.

    mu and delta should have a leading batch dimension of size M.

    :param mu: Mean activities of n populations.
    :type mu: tf.Tensor (M x n)
    :param delta: Variance of activities of n populations.
    :type delta: tf.Tensor (M x n)
    :param num_pts: Number of points in the gaussian quadrature.
    :type num_pts: int
    :return: Gaussian integral of phi(z).
    :rtype: tf.Tensor (M x n)
    """
    mu = mu[:,:,:,None]
    delta = delta[:,:,:,None]
    g_pts, g_weights = np.polynomial.hermite.hermgauss(num_pts)
    g_pts = (np.sqrt(2.0) * g_pts)[None, None, None, :]
    g_weights = (g_weights / np.sqrt(np.pi))[None, None, :, None]

    r = mu + delta * g_pts
    r[r<0.] = 0.
    phi = r**2
    m = np.matmul(phi, g_weights)[:,:,:,0]

    return m


def IntegralPhiSq(mu, delta, num_pts=50):
    """Calculates gaussian integral of :math:`\\phi^2(z)` given mu and delta.

    mu and delta should have a leading batch dimension of size M.

    :param mu: Mean activities of n populations.
    :type mu: Tensor (M x n)
    :param delta: Variance of activities of n populations.
    :type delta: Tensor (M x n)
    :param num_pts: Number of points in the gaussian quadrature.
    :type num_pts: int
    :return: Gaussian integral of phi^2(z).
    :rtype: tf.Tensor (M x n)
    """
    mu = mu[:, :, :, None]
    delta = delta[:, :, :, None]
    g_pts, g_weights = np.polynomial.hermite.hermgauss(num_pts)
    g_pts = (np.sqrt(2.0) * g_pts)[None, None, None, :]
    g_weights = (g_weights / np.sqrt(np.pi))[None, None, :, None]
    
    r = mu + delta * g_pts
    r[r<0.] = 0.
    phi = r**2
    phi_sq = phi**2
    v = np.matmul(phi_sq, g_weights)[:,:,:,0]
    return v

def mu_eq(w, m, f, h):
    """Consistency equation for mu.

    :param w: Model weight means.
    :type w: Tensor (M x n, n)
    :param m: Mean activities.
    :type m: Tensor (M x n)
    :param f: Relative proportions of neurons.
    :type f: np.ndarray (n,)
    :param h: Model input mean.
    :type h: Tensor (M x n)
    :return: mu
    :rtype: Tensor (M x n)
    """
    fm = f * m

    mu = np.matmul(w, fm) + h
    return mu


def delta_eq(sigma_sq, v, f, lambda_sq):
    """Consistency equation for delta.

    :param sigma_sq: Square of sigma.
    :type w: Tensor (M x n, n)
    :param v: Activity variance.
    :type v: Tensor (M x n)
    :type f: np.ndarray (n,)
    :param h: Model input mean.
    :param lambda_sq: Square of lambda.
    :type lambda_sq: Tensor (M x n)
    :return: delta
    :rtype: Tensor (M x n)
    """
    fv = f * v

    delta = np.matmul(sigma_sq, fv) + lambda_sq
    return delta


def mean_field_EI(z, traj=False):
    n = 2  # Number of populations.
    num_gauss_pts = 50  # Number of points used to estimate Gaussian integrals.
    M = z.shape[0]
    W = np.reshape(z[:,:4], (M, 2, 2))
    #sigma = 0.2*np.ones((1,2,2))
    sigma = np.reshape(z[:,4:], (M, 2, 2))

    # Step size for langevin dynamics consistency equation solver.
    step = 0.5
    # Number of steps of solver.  Should check that solver converges over parameter support
    num_iters = 100


    cs = [0.0, 0.5, 1.0]
    num_cs = len(cs)

    f = np.array([[[0.5], [0.5]]])  # Relative proportions of populations.
    h = np.zeros((1, n, num_cs))
    for i in range(num_cs):
        h[0, :, i] = 0.8 + 0.8 * cs[i]

    lam = 0.5 * np.ones((1, 2, 1))  # Noise level (variance of the input).
    lambda_sq = lam ** 2

    # Bounds for mean field parameters
    mu_min = 0.0
    mu_max = 1000.0
    delta_min = 0.0
    delta_max = 1000.0

    # Mean field activity averages (mu) and variances (delta).
    mu = 0.1 * np.random.rand(1, n, num_cs)
    delta = 0.1 * np.random.rand(1, n, num_cs)
    if traj:
        mus = np.zeros((num_iters+1, M, n, num_cs))
        deltas = np.zeros((num_iters+1, M, n, num_cs))
        mus[0] = mu 
        deltas[0] = delta
    # Solve equations.
    for i in range(1, num_iters+1):
        m = IntegralPhi(mu, delta, num_pts=num_gauss_pts)
        v = IntegralPhiSq(mu, delta, num_pts=num_gauss_pts)
        mu_next = mu_eq(W, m, f, h)
        delta_next = delta_eq(sigma ** 2, v, f, lambda_sq)
        mu = np.clip((1 - step) * mu + step * mu_next, mu_min, mu_max)
        delta = np.clip(
            (1 - step) * delta + step * delta_next, delta_min, delta_max
        )
        if traj:
            mus[i] = mu
            deltas[i] = delta

    # Emergent property statistics are the first moments of MF variables mu and delta.
    mu = np.reshape(mu, (M, n * num_cs))
    delta = np.reshape(delta, (M, n * num_cs))
    T_x = np.concatenate((mu, delta), axis=1)
    if traj:
        return T_x, mus, deltas
    else:
        return T_x



def mean_field_4n(z, traj=False):
    n = 4  # Number of populations.
    num_gauss_pts = 50  # Number of points used to estimate Gaussian integrals.
    M = z.shape[0]
    W = np.reshape(z[:,:16], (M, n, n))
    sigma = 0.2*np.ones((1,n,n))
    #sigma = np.reshape(z[:,16:], (M, n, n))

    # Step size for langevin dynamics consistency equation solver.
    step = 0.5
    # Number of steps of solver.  Should check that solver converges over parameter support
    num_iters = 100


    cs = [0, 6, 12, 25, 50, 100]
    num_cs = len(cs)

    f = np.array([[[0.3], [0.7*0.4], [0.7*0.3], [0.7*0.3]]])  # Relative proportions of populations.
    h = np.zeros((1, n, num_cs))
    for i in range(num_cs):
        h[0, :, i] = 0.2 + 0.01 * cs[i]

    lam = 0.5 * np.ones((1, n, 1))  # Noise level (variance of the input).
    lambda_sq = lam ** 2

    # Bounds for mean field parameters
    mu_min = 0.0
    mu_max = 1000.0
    delta_min = 0.0
    delta_max = 1000.0

    # Mean field activity averages (mu) and variances (delta).
    mu = 0.1 * np.random.rand(1, n, num_cs)
    delta = 0.1 * np.random.rand(1, n, num_cs)
    if traj:
        mus = np.zeros((num_iters+1, M, n, num_cs))
        deltas = np.zeros((num_iters+1, M, n, num_cs))
        mus[0] = mu 
        deltas[0] = delta
    # Solve equations.
    for i in range(1, num_iters+1):
        m = IntegralPhi(mu, delta, num_pts=num_gauss_pts)
        v = IntegralPhiSq(mu, delta, num_pts=num_gauss_pts)
        mu_next = mu_eq(W, m, f, h)
        delta_next = delta_eq(sigma ** 2, v, f, lambda_sq)
        mu = np.clip((1 - step) * mu + step * mu_next, mu_min, mu_max)
        delta = np.clip(
            (1 - step) * delta + step * delta_next, delta_min, delta_max
        )
        if traj:
            mus[i] = mu
            deltas[i] = delta

    # Emergent property statistics are the first moments of MF variables mu and delta.
    mu = np.reshape(np.delete(mu, 1, 1), (M, (n-1) * num_cs))
    delta = np.reshape(np.delete(delta, 1, 1), (M, (n-1) * num_cs))
    T_x = np.concatenate((mu, delta), axis=1)
    if traj:
        return T_x, mus, deltas
    else:
        return T_x

