import numpy as np
from scipy.integrate import odeint
from scipy.spatial import cKDTree as KDTree
from scipy import stats
import matplotlib.pyplot as plt
## dinamic equations Rossler
def Rossler(X, t, a=0.15, b=0.2, c=10):
    x,y,z = X
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + x*z - c*z
    return (x_dot, y_dot, z_dot)

## Rossler attractor
def Rossler_attractor(N,dt, a=0.15, b=0.2, c=10):
    tmax = dt*N
    t = np.linspace(0, tmax, N)

    # The initial conditions
    x0, y0, z0 = (-3.2916983, -1.42162302, 0.02197593)

    f = odeint(Rossler, (x0, y0, z0), t, args=(a, b, c))
    x, y, z = f.T
    return x,y,z

## embedding function of a time series
def embedding(x,m,tau):
    M=len(x)-(m-1)*tau
    X_matr=np.zeros((M,m))
    for i in range(0,M,1):
        for l in range (0,m,1):
            X_matr[i][l]=x[i+l*tau]
    return X_matr

## function to derivate a time series
def der(x):
    dx=np.zeros(len(x))
    for i in range(len(x)-1):
        dx[i]=x[i+1]-x[i]
    return dx

## function to compute mutual information of 2 variables x,y
def mi(x, y, bins=64):

    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    # Convert frequencies into probabilities.  Also, in the limit
    # p -> 0, p*log(p) is 0.  We need to take out those.
    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    # Calculate the corresponding Shannon entropies.
    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y

def time_mi(x, maxtau=1000, bins=64):

    """Return the time-delayed mutual information of x_i.
    Parameters
    ----------
    x : array
        1-D real time series of length N.
    maxtau : int, optional (default = min(N, 1000))
        Return the mutual information only up to this time delay.
    bins : int
        Number of bins to use while calculating the histogram.

    Returns
    -------
    ii : array
        Array with the time-delayed mutual information up to maxtau.
    """
    N = len(x)
    maxtau = min(N, maxtau)

    ii = np.empty(maxtau)
    tau_arr = np.empty(maxtau)

    ii[0] = mi(x, x, bins)
    tau_arr[0]=0

    for tau in range(1, maxtau):
        ii[tau] = mi(x[:-tau], x[tau:], bins)
        tau_arr[tau]=tau

    return ii,tau_arr

##autocorrelation function
def corr(x,y,dim):
    c=(dim*np.sum(x*y)-np.sum(x)*np.sum(y))/(np.sqrt((dim*np.sum(x**2)-(np.sum(x))**2)*(dim*np.sum(y**2)-(np.sum(y))**2)))
    return c

##autocorrelation function vs tau
def corr_arr(x,maxtau=1000):
    c= np.empty(maxtau)
    tau_arr=np.empty(maxtau)
    c[0]=1
    tau_arr[0]=0

    for tau in range(1, maxtau):
        c[tau] = corr(x[:-tau], x[tau:],len(x))
        tau_arr[tau]=tau
    return c,tau_arr

##distance functions
def euclidean_dist(x, y):
    return np.sqrt(np.array(list(map(np.sum, (x - y) ** 2))))

def chebyshev_dist(x, y):
    return np.array(list(map(np.max, np.abs(x - y))))

def neighbors(y, metric='chebyshev', window=0, maxnum=None):
    """Find nearest neighbors of all points in the given array.

    Parameters
    ----------
    y : ndarray
        N-dimensional array containing time-delayed vectors.
    window : int, optional (default = 0)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
        Lyapunov exponents and the correlation dimension.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

    Returns
    -------
    index : array
        Array containing indices of near neighbors.
    dist : array
        Array containing near neighbor distances.
    """
    if metric == 'euclidean':
        p = 2
    elif metric == 'chebyshev':
        p = np.inf

    tree = KDTree(y)
    n = len(y)

    if not maxnum:
        maxnum = (window + 1) + 1 + (window + 1)
    else:
        maxnum = max(1, maxnum)

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        for k in range(2, maxnum + 2):
            dist, index = tree.query(x, k=k, p=p)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid):
                dists[i] = dist[valid][0]
                indices[i] = index[valid][0]
                break

            if k == (maxnum + 1):
                raise Exception('Could not find any near neighbor with a'
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    return np.squeeze(indices), np.squeeze(dists)


def parallel_map(func, values, args=tuple(), kwargs=dict()):
    return np.asarray([func(value, *args, **kwargs) for value in values])

##fnn
def fnn(d, x, r,tau=1, metric='euclidean', window=10,
         maxnum=None):
    """Return fraction of false nearest neighbors for a single d.
    """
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = embedding(x[:-tau],d,tau)
    y2 = embedding(x, d + 1, tau)

    # Find near neighbors in dimension d.
    index, dist = neighbors(y1, metric=metric, window=window,
                                  maxnum=maxnum)

    # Find all potential false neighbors using Kennel et al.'s tests.
    f2 = euclidean_dist(y2, y2[index]) < np.std(x)/r
    f1= euclidean_dist(y2, y2[index])/ dist > r
    f3=np.sum(f1 & f2)/np.sum(f2)

    return f3

## lyapunov exponent fit
def mean_lyapunov_exp(x, max_l=500, window=10, metric='euclidean', maxnum=None):
    """Estimate the maximum Lyapunov exponent.
    """
    index, dist = neighbors(x, metric=metric, window=window,
                                  maxnum=maxnum)
    m = len(x)
    max_l = min(m - window - 1, max_l)

    d = np.empty(max_l)
    t_arr = np.empty(max_l)
    d[0] = np.mean(np.log(dist))
    t_arr[0]=0

    for t in range(1, max_l):
        t1 = np.arange(t, m)
        t2 = index[:-t] + t

        # Sometimes the nearest point would be farther than (m - maxt)
        # in time.  Such trajectories needs to be omitted.
        valid = t2 < m
        t1, t2 = t1[valid], t2[valid]

        d[t] = np.mean(np.log(euclidean_dist(x[t1], x[t2])))
        t_arr[t]=t
    return d,t_arr


def grassberg_procaccia(X,sigma):

    n_points = len(X)

    # Timeseries standard deviation
    data_std = sigma

    # Generate a series of r distances evenly spaced in log scale, these are
    # generated starting from the timeseries of scalars standard deviation.
    # The r distance is a scalar used to find the fraction of points in phase space for which
    # the euclidean distance between them is smaller than r

    r_vals = np.linspace(0.1 * data_std, 0.7 * data_std, 30)


    distances = np.zeros(shape=(n_points,n_points))
    r_matrix_base = np.zeros(shape=(n_points,n_points))

    # Euclidean distance of points in phase space
    for i in range(n_points):
        for j in range(i,n_points):
            distances[i][j] = np.linalg.norm(X[i]-X[j])
            r_matrix_base[i][j] = 1

    # Correlation sum
    C_r = []
    for r in r_vals:
        r_matrix = r_matrix_base*r
        heavi_matrix = np.heaviside( r_matrix - distances,0)
        corr_sum = (2/float(n_points*(n_points-1)))*np.sum(heavi_matrix)
        C_r.append(corr_sum)


    return np.array(C_r),r_vals








