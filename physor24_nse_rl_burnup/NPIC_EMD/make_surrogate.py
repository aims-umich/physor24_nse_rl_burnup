import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_kq_filled, train_split

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence
from skopt.utils import use_named_args

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

t = 4
parmsearch = 30

def log(time, train_samples, test_samples, learning_rate, num_layers, 
        nodes_per_layer, r2, mae, mse):
    f = Path("models/t%s"%(str(time)) + "/hyperopt_log.dat")
    l = "%i, %i, %.4E, %i, %i, %.4E, %.4E, %.4E, %s"%(train_samples, test_samples, learning_rate, num_layers, 
            nodes_per_layer, r2, mae, mse, datetime.datetime.now())
    l = ",".join([a.ljust(25) for a in l.split(",")]) + "\n"
    if f.exists():
        with open(f, "a") as g:
            g.write(l)
    else:
        with open(f, "w") as g:
            hdr = "train samples, test samples, learning_rate, "+\
                "num_layers, nodes_per_layer, r2, mae, "+\
                "mse, time"
            hdr = ",".join([a.ljust(25) for a in hdr.split(",")]) + "\n"
            g.write(hdr)
            g.write(l)

def opt_model(time, Xtrain, Xvalid, ytrain, yvalid, N,
              b_learning_rate = [1e-4, 1e-3],
              b_num_layers = [10, 20],
              b_nodes_per_layer = [45, 100],
              test_size = 0.1):

    #function for optimization
    def calc_loss(learning_rate, num_layers, nodes_per_layer):
        model = create_model(Xtrain, ytrain, Xvalid, yvalid, learning_rate, num_layers,
                nodes_per_layer)
        ynn = model.predict(Xvalid)

        ynn.flatten()
        ytest.flatten()

        r2 = r2_score(yvalid, ynn)
        mse = ((yvalid- ynn)**2).mean()
        mae = np.abs(yvalid - ynn).mean()

        log(time, Xtrain.shape[0], Xtest.shape[0], learning_rate, num_layers,
                nodes_per_layer, r2, mae, mse)

        return mae

    #perform search
    #Define optimization objects
    dim_learning_rate = Real(low = b_learning_rate[0], high = b_learning_rate[1],
            prior = "uniform", name = "learning_rate")
    dim_num_layers = Integer(low = b_num_layers[0], high = b_num_layers[1],
            name = "num_layers")
    dim_nodes_per_layer = Integer(low = b_nodes_per_layer[0], high = b_nodes_per_layer[1],
            name = "nodes_per_layer")

    dims = [dim_learning_rate, dim_num_layers, dim_nodes_per_layer]

    @use_named_args(dimensions = dims)
    def fitness(learning_rate, num_layers, nodes_per_layer):
        return calc_loss(learning_rate, num_layers, nodes_per_layer)

    initial_guess = [np.mean(b_learning_rate),
                     int(np.mean(b_num_layers)),
                     int(np.mean(b_nodes_per_layer))]
    #load in previous guesses
    try:
        lg = pd.read_csv("models/t%i/hyperopt_log.dat"%time)
        Xdones = [[float(a[0]), int(a[1]), int(a[2])] for a in lg.iloc[:, 2:5].values]
        ydones = lg.iloc[:, 6].values

        search_result = gp_minimize(func = fitness,
                                    dimensions = dims,
                                    n_initial_points = 0,
                                    acq_func = "EI",
                                    n_calls = N,
                                    x0 = Xdones,
                                    y0 = ydones,
                                    verbose = True)
    except:
        initial_guess = [np.mean(b_learning_rate),
                         int(np.mean(b_num_layers)),
                         int(np.mean(b_nodes_per_layer))]
        search_result = gp_minimize(func = fitness,
                                    dimensions = dims,
                                    acq_func = "EI",
                                    n_calls = N,
                                    x0 = initial_guess,
                                    verbose = True)
    return search_result.x

def create_model(Xtrain, ytrain, Xvalid, yvalid, learning_rate, num_layers, nodes_per_layer):
    stp = EarlyStopping(monitor = "val_loss",
                        patience = 500,
                        restore_best_weights = True)

    model = Sequential()
    model.add(Dense(Xtrain.shape[1], kernel_initializer = "normal", input_dim = Xtrain.shape[1]))

    for i in range(1, num_layers):
        model.add(Dense(nodes_per_layer, kernel_initializer = "normal",
                activation = "relu"))

    model.add(Dense(ytrain.shape[1], kernel_initializer = "normal",
        activation = "linear"))

    model.compile(loss = "mse",
            optimizer = Adam(learning_rate),
            metrics = ["mse"])

    model.fit(Xtrain, ytrain, epochs = 50000, verbose = 1, validation_data = (Xvalid, yvalid), callbacks = [stp])
    return model




#initial data load
d = get_kq_filled(t, "20230301")

#scaling quantities
k_bnds = [0.88, 1.07]
Q_bnds = [0.15, 0.19]
theta_bnds = [0, 360]

#data splitting
train, remain = train_split(d, tst_frac = 0.3)
test, valid = train_split(remain, tst_frac = 0.5)

response_names = ["k"] + ["Q" + str(i) for i in range(1, 7)]
predictor_names = ["theta" + str(i) for i in range(1, 7)]

#data reformating
Xtrain_u = train[predictor_names].values
ytrain_u = train[response_names].values

Xvalid_u = valid[predictor_names].values
yvalid_u = valid[response_names].values

Xtest_u = test[predictor_names].values
ytest_u = test[response_names].values


#data recaling
Xtrain = Xtrain_u / 360
Xvalid = Xvalid_u / 360
Xtest = Xtest_u / 360

ytrain = np.zeros_like(ytrain_u)
ytrain[:,0] = (ytrain_u[:,0] - k_bnds[0]) / (k_bnds[1] - k_bnds[0])
ytrain[:,1:] = (ytrain_u[:,1:] - Q_bnds[0]) / (Q_bnds[1] - Q_bnds[0])

yvalid = np.zeros_like(yvalid_u)
yvalid[:,0] = (yvalid_u[:,0] - k_bnds[0]) / (k_bnds[1] - k_bnds[0])
yvalid[:,1:] = (yvalid_u[:,1:] - Q_bnds[0]) / (Q_bnds[1] - Q_bnds[0])

ytest = np.zeros_like(ytest_u)
ytest[:,0] = (ytest_u[:,0] - k_bnds[0]) / (k_bnds[1] - k_bnds[0])
ytest[:,1:] = (ytest_u[:,1:] - Q_bnds[0]) / (Q_bnds[1] - Q_bnds[0])

#find optimal model parameters
if True:
    learning_rate, num_layers, nodes_per_layer =  opt_model(t, Xtrain, Xvalid, ytrain, yvalid, parmsearch,
                                                                    b_learning_rate = [1e-4, 1e-2],
                                                                    b_num_layers = [5, 20],
                                                                    b_nodes_per_layer = [5, 200])
else:
    learning_rate, num_layers, nodes_per_layer = 1.75e-3, 5, 200

#train model
model = create_model(Xtrain, ytrain, Xvalid, yvalid, learning_rate, num_layers, nodes_per_layer)
model.save(Path("models/t" + str(t) + "/s"))

m = load_model(Path("models/t" + str(t) + "/s"))

#model performance evaluation
perf = open(Path("models/t%i/performance.dat"%t), "w")
perf.write("learning_rate " + str(learning_rate) + "\n")
perf.write("num_layers " + str(num_layers) + "\n")
perf.write("nodes_per_layer " + str(nodes_per_layer) + "\n")

# --- training evaluation
ytrain_pred_scaled = m.predict(Xtrain)
ytrain_pred = np.zeros_like(ytrain_pred_scaled)
ytrain_pred[:,0] = ytrain_pred_scaled[:,0]*(k_bnds[1] - k_bnds[0]) + k_bnds[0]
ytrain_pred[:,1:] = ytrain_pred_scaled[:,1:]*(Q_bnds[1] - Q_bnds[0]) + Q_bnds[0]
train_errs = ytrain_pred - ytrain_u
perf.write("TRAIN\n======\n")
for i in range(len(response_names)):
    err = train_errs[:,i]
    mae = np.abs(err).mean()
    mse = (err**2).mean()
    R2 = r2_score(ytrain_u[:,i], ytrain_pred[:,i])
    perf.write(response_names[i] + "\n")
    perf.write("MAE " + str(mae) + "\n")
    perf.write("MSE " + str(mse) + "\n")
    perf.write("R2 " + str(R2) + "\n\n")

# --- testing evaluation
ytest_pred_scaled = m.predict(Xtest)
ytest_pred = np.zeros_like(ytest_pred_scaled)
ytest_pred[:,0] = ytest_pred_scaled[:,0]*(k_bnds[1] - k_bnds[0]) + k_bnds[0]
ytest_pred[:,1:] = ytest_pred_scaled[:,1:]*(Q_bnds[1] - Q_bnds[0]) + Q_bnds[0]
test_errs = ytest_pred - ytest_u
perf.write("TEST\n======\n")
for i in range(len(response_names)):
    err = test_errs[:,i]
    mae = np.abs(err).mean()
    mse = (err**2).mean()
    R2 = r2_score(ytest_u[:,i], ytest_pred[:,i])
    perf.write(response_names[i] + "\n")
    perf.write("MAE " + str(mae) + "\n")
    perf.write("MSE " + str(mse) + "\n")
    perf.write("R2 " + str(R2) + "\n\n")
perf.close()

