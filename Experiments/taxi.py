import sys
sys.path.insert(0, '../')
from AnomalyDetection import _3DTPCA_AnomalyDetection, LSTMAutoEncoder_AnomalyDetection, PCA_AnomalyDetection, get_roc_curve_values

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from sklearn.preprocessing import StandardScaler

import pickle

from time import time

N = 2191
l, m = 22, 22
scaler = StandardScaler()
data = pickle.load(open("data/smol_adjmat.pkl", "rb"))
scaled = scaler.fit_transform(data.reshape((-1,1))).reshape((N,l,m))

p = 7
N_test = 365
N_train = N - N_test
actual = np.zeros((N_test,),dtype=int)
actual[72:] = 1
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

print("Fitting 3DTPCA..")
_3dtpca = _3DTPCA_AnomalyDetection(scaled, N_train, N_test, p, economy_mode=True, debug=True)
_3dtpca.fit(0.90)

print("Fitting LSTM Autoencoder...")
lstm = LSTMAutoEncoder_AnomalyDetection(scaled, N_train, N_test, p, [
    LSTM(1024, activation="relu", input_shape=(p,l*m), return_sequences=True),
    LSTM(256, activation="relu", return_sequences=True),
    LSTM(64, activation="relu", return_sequences=False),
    RepeatVector(p),
    LSTM(64, activation="relu", return_sequences=True),
    LSTM(256, activation="relu", return_sequences=True),
    LSTM(1024, activation="relu", return_sequences=True),
    TimeDistributed(Dense(l*m))
])
lstm.fit(epochs=20)

print("Fitting PCA...")
pca = PCA_AnomalyDetection(data, N_train, N_test)
pca.fit(20)

print("Saving Models...")
with open("results/taxi_3dtpca.pkl", "wb") as f:
    pickle.dump(_3dtpca, f)
with open("results/taxi_lstm.pkl", "wb") as f:
    pickle.dump(lstm, f)
with open("results/taxi_pca.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("results/taxi_actual.pkl", "wb") as f:
    pickle.dump(actual, f)

print("Testing Speed")
_3dtpca_times = [] 
lstm_times = [] 
pca_times = [] 
for i in range(20):

    print("Running trial", i)

    start = time()
    _3dtpca.fit(0.90)
    end = time()
    _3dtpca_times.append(end-start)

    start = time()
    lstm.fit(epochs=20)
    end = time()
    lstm_times.append(end-start)

    start = time()
    pca.fit(50)
    end = time()
    pca_times.append(end-start)

times_df = pd.DataFrame()
times_df["3DTPCA"] = _3dtpca_times
times_df["LSTM"] = lstm_times
times_df["PCA"] = pca_times
print(times_df)
print(times_df.describe())
times_df.to_csv("results/taxi_times.csv", index=False)