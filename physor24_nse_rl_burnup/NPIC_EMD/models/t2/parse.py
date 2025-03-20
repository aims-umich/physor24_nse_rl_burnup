import numpy as np

with open("performance.dat", "r") as f:
    c = f.readlines()[48:]

maes = []
mses = []
r2s = []

d = {"MAE" : maes,
    "MSE" : mses,
    "R2" : r2s}

for l in c:
    try:
        a=l.split()
        d[a[0]].append(float(a[1]))
    except:
        pass

for k, v in d.items():
    print(k, np.mean(v))

