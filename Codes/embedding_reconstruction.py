import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema



def Rossler(X, t, a=0.2, b=0.2, c=6.3):
    x,y,z = X
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + x*z - c*z
    return (x_dot, y_dot, z_dot)

def Rossler_attractor(a=0.2, b=0.2, c=6.3, N=50000):
    tmax = 0.01*N
    t = np.linspace(0, tmax, N)

    # The initial conditions
    x0, y0, z0 = (1,1,1)

    f = odeint(Rossler, (x0, y0, z0), t, args=(a, b, c))
    x, y, z = f.T
    return x,y,z

def embedding(x,m,tau,M):
    X_matr=np.zeros((M,m))
    for i in range(0,M,1):
        for l in range (0,m,1):
            X_matr[i][l]=x[i+l*tau]
    return X_matr

def der(x):
    dx=np.zeros(len(x))
    for i in range(len(x)-1):
        dx[i]=x[i+1]-x[i]
    return dx

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

def dmi(x, maxtau=1000, bins=64):

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

def corr(x,y,dim):
    c=(dim*np.sum(x*y)-np.sum(x)*np.sum(y))/(np.sqrt((dim*np.sum(x**2)-(np.sum(x))**2)*(dim*np.sum(y**2)-(np.sum(y))**2)))
    return c

def corr_arr(x,maxtau=1000):
    c= np.empty(maxtau)
    tau_arr=np.empty(maxtau)
    c[0]=1
    tau_arr[0]=0

    for tau in range(1, maxtau):
        c[tau] = corr(x[:-tau], x[tau:],len(x))
        tau_arr[tau]=tau
    return c,tau_arr

if __name__ == "__main__":
    #serie temporali
    N=50000
    xs, ys, zs = Rossler_attractor()
    m=3
    tau=100
    M=len(xs)-(m-1)*tau

    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    plt.tick_params(labelsize=15)
    ax.set_title('Rossler Attractor', fontsize=15)

    fig = plt.figure(2)
    X_matr=embedding(xs,m,tau,M)
    X_matr=np.array(X_matr)
    #phase reconstruction
    plt.errorbar(X_matr[:,0],X_matr[:,1],lw=0.5)

    fig = plt.figure(3)
    #mutual information
    I,tau=dmi(xs)
    plt.errorbar(tau,I)
    #tau=np.argsort(I)[0]
    taumax=argrelextrema(I, np.less, order=10)
    print(taumax)

    fig = plt.figure(4)
    #correlation
    corr,tau=corr_arr(xs)
    plt.errorbar(tau,corr)
    taumax=argrelextrema(corr, np.less, order=10)
    print(taumax)

    plt.show()