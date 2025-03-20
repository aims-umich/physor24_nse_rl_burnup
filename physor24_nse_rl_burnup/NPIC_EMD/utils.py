import numpy as np
from pathlib import Path
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt

def get_k_unpert(time, date):
    """time is int, 0,2 or 4. date is str of date"""
    datapath = (Path(os.path.realpath(__file__)).parent / Path("data/" + date)).resolve()

    k_unpert = pd.read_csv(datapath / Path("k_unperturbed.dat"), index_col = 0)
    return k_unpert.loc["YM%i"%time]["k"]

def get_jm(time, date):
    """time is int, 0,2 or 4. date is str of date"""
    datapath = (Path(os.path.realpath(__file__)).parent / Path("data/" + date)).resolve()
    d = pd.read_csv(datapath / Path("j_%iYR.dat"%time))
    return d

def get_jm_adj(time, date):
    d = get_jm(time, date)
    d["theta"] = (180 - d["theta"]) % 360
    norm = np.trapz(d["j"], d["theta"])
    d["j"] /= norm
    d = d.drop("j_abs_uncert", axis = 1)
    d2 = d.copy()
    d2["theta"] += 360
    d3 = d.copy()
    d3["theta"] -= 360
    d = pd.concat([d, d2, d3])
    return d.sort_values("theta")

def get_kq(time, date):
    """time is int, 0,2 or 4. date is str of date"""
    datapath = (Path(os.path.realpath(__file__)).parent / Path("data/" + date)).resolve()
    d = pd.read_csv(datapath / Path("kq_%iYR.dat"%time), index_col = 0)
    return d

def sample_mult(df):
    ths = ["theta" + str(i) for i in range(1, 7)]
    Qs = ["Q" + str(i) for i in range(1, 7)]

    all_data = {}
    for n in df.index:
        for i in range(6):
            temp = copy.deepcopy(df.loc[n])
            temp[ths] = np.roll(df.loc[n][ths].values, i)
            temp[Qs] = np.roll(df.loc[n][Qs].values, i)
            all_data[n + "_s" + ("%i"%(i+1)).zfill(2)] =  temp
        for i in range(6):
            temp = copy.deepcopy(df.loc[n])
            temp[ths] = np.roll(df.loc[n][ths].values[::-1], i)
            temp[Qs] = np.roll(df.loc[n][Qs].values[::-1], i)
            all_data[n + "_s" + ("%i"%(6+i+1)).zfill(2)] =  temp
    return pd.DataFrame.from_dict(all_data, orient = "index")


def train_split(data, tst_frac = False, tst_num = False):
    """splits into training and testing while not letting
    equivalent samples end up on different sides of the divide
    tst_frac is fraction of testing
    tst_num is number of CALCULATIONS to include in test set"""

    numuniq = np.unique([a[:9] for a in data.index]).size

    if tst_frac:
        tst_num = round(numuniq*tst_frac)
        
    selected = np.random.choice(numuniq, tst_num, replace = False)

    if numuniq == data.shape[0]:
        mask = np.zeros(data.shape[0], dtype = bool)
        mask[selected] = True
        return data.iloc[~mask], data.iloc[mask]
    else:
        expanded_idxs = np.array([[12*i + j for j in range(12)] for i in selected]).flatten()
        mask = np.zeros(data.shape[0], dtype = bool)
        mask[expanded_idxs] = True
        return data.iloc[~mask], data.iloc[mask]

def include_lambda(df):
    """
    inplace add lambda calculation and std-dev to a pandas dataframe
    """
    df["lambda"] = 1/df["k"]

def get_kq_filled(time, date): #MAJDI THIS IS THE ONE THAT MATTERS
    d = sample_mult(get_kq(time, date)) #USE IT LIKE: d = get_kq_filled(0, "20230301")
    include_lambda(d)
    return d
    
