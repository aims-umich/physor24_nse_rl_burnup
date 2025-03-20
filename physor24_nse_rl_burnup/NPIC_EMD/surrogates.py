from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path

class Surr:
    def __init__(self, t):
        """t is an int of 0, 2 or 4 depending on what year the surrogates are made for"""
        self.m = load_model(Path(__file__).resolve().parent / Path("models/t%s/s"%str(t)))
        self.k_bnds = [0.88, 1.07]
        self.Q_bnds = [0.15, 0.19]

    def predict(self, X):
        """
        Input:
            X: numpy array of shape (# of input calculations, 6) which holds control drum angles
               in degrees for prediction
        Output:
            k: estimated core criticality
            Qs: numpy array of hexant powers, as fraction
        """
        shp = X.shape
        if len(shp) == 1:
            Xin = np.zeros((1, X.size))
        else:
            Xin = np.zeros_like(X)
        Xin[:, :] = X / 360
        y = self.m.predict(Xin)
        y[:,0] = y[:,0]*(self.k_bnds[1] - self.k_bnds[0]) + self.k_bnds[0]
        y[:,1:] = y[:,1:]*(self.Q_bnds[1] - self.Q_bnds[0]) + self.Q_bnds[0]
        k = y[:, 0]
        Qs = y[:,1:]/ y[:, 1:].sum(1).reshape(-1, 1)
        return k, Qs

if __name__ == "__main__":
    a = Surr(2)
    print(a.predict(np.random.uniform(0, 360, (1, 6))))




