import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad

datadate = "20221206"

#coating angle is 110 degrees
d = get_kq_filled(0, datadate)

#%%

def calc_Q(theta, time): #coul be sped up
    """
    function to calculate Q based on drum and rotation angle as 
    given in Equation 78
    """
    d1, d2 = calc_ints(theta)
    jd = get_jm_adj(time, datadate)
    f = interp1d(jd["theta"], jd["j"])
    f2 = lambda beta : f(beta)**2
    r1 = np.linspace(d1[0], d1[1], 400)
    t1 = np.trapz(f2(r1).flatten(), r1)
    r2 = np.linspace(d2[0], d2[1], 400)
    t2 = np.trapz(f2(r2).flatten(), r2)
    return t1 - t2

def get_X(train, time):
    """create X matrix"""
    I = train.shape[0]
    N = 6
    X = np.zeros((I, N*(N+1)))
    for i in range(I):
        gammastar = np.ones(N+1)
        for n in range(0, N):
            col = "theta" + str(n+1)
            gammastar[n+1] = albedo(train[col].iloc[i], time)
        for n in range(N):
            col = "theta" + str(n + 1)
            qeval = calc_Q(train.iloc[i][col], time)
            X[i, (N+1)*n:(N+1)*(n+1)] = gammastar * qeval 
    return X

def get_X_fo(train, time):
    """create X matrix"""
    I = train.shape[0]
    N = 6
    X = np.zeros((I, N))
    for i in range(I):
        for n in range(N):
            thetatag = "theta" + str(n + 1)
            X[i, n] = calc_Q(train.iloc[i][thetatag], time)
    return X

def calc_ints(da):
    """
    find intersection of of relevant domains
    """
    
    if da > 180:
        da =  da - 360
    elif da < -180:
        da = 360 + da

    if np.isclose(da, 0):
        return np.zeros(2), np.zeros(2)

    domain = np.array([-0.5, 0.5])*110
    
    if 110 < abs(da):
        return domain, domain + da
    elif da > 0:
        d1 = np.array([-55, da - 55])
        d2 = np.array([55, da + 55])
        return d1, d2
    elif da < 0:
        d1 = np.array([da + 55, 55])
        d2 = np.array([da - 55, -55])
        return d1, d2
    else:
        raise Exception("da is 0")

def albedo(theta, time): #can speedup if needed
    jd = get_jm_adj(time, datadate)
    lb = theta - 55
    ub = theta + 55
    
    j = jd["j"]
    th = jd["theta"]
    
    idxs = np.where( (th < ub)*(th > lb))[0]
    
    bulk = np.trapz(j.iloc[idxs], th.iloc[idxs])
    
    bot = j.iloc[idxs[0] - 1] + (j.iloc[idxs[0]] - j.iloc[idxs[0] - 1])  \
        /(th.iloc[idxs[0]] - th.iloc[idxs[0] - 1])  \
        *(lb - th.iloc[idxs[0] - 1])
    bot_trap = 0.5*(th.iloc[idxs[0]] - lb)*(bot + j.iloc[idxs[0]])

    top = j.iloc[idxs[-1]] + (j.iloc[idxs[-1] + 1] - j.iloc[idxs[-1]])  \
        /(th.iloc[idxs[-1] + 1] - th.iloc[idxs[-1]])  \
        *(ub - th.iloc[idxs[-1]])
    top_trap = 0.5*(ub - th.iloc[idxs[-1]])*(top + j.iloc[idxs[-1]])
    return  bulk + top_trap + bot_trap
    
def fit(train, time, nom):
    """find alpha values for model"""
    X = get_X(train, time)
    y = (train["lambda"] - 1/nom).values
    
    alpha = np.linalg.pinv(X)@y
    return alpha

def predict(test, train, time):
    nom = get_k_unpert(time, datadate)
    
    alpha = fit(train, time, nom)
    pred_react = get_X(test, time)@alpha
    
    pred_k = 1/(pred_react + 1/nom)
    return pred_k
    
def test_train_eval(test, train, time):
    k_hat = predict(test, train, time)
    k_real = test["k"]
    
    errs = k_hat - k_real
    print(errs)

if __name__ == "__main__":
    t1, t2 = train_split(d, tst_num = 10)
    print(t1.shape)
    print(t2.shape)
    #ks = test_train_eval(d.iloc[:100], d.iloc[:100], 0)
    

