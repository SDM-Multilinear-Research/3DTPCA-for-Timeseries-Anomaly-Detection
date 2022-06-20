import sys
sys.path.insert(0, '../')
from AnomalyDetection import _3DTPCA_AnomalyDetection, LSTMAutoEncoder_AnomalyDetection, PCA_AnomalyDetection

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time

import pickle

N = 4000
l, m = 3, 3
data = np.zeros((N,l,m))
for i in range(N):
    data[i] = np.loadtxt(f"data/GeneratedLTAR/Observation{i}.txt")

p = 5
N_test = 1500
N_train = N - N_test
actual = np.zeros((N_test,),dtype=int)
actual[500:] = 1
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

print("Fitting 3DTPCA..")
_3dtpca = _3DTPCA_AnomalyDetection(data, N_train, N_test, p, economy_mode=True, debug=True)
_3dtpca.fit(0.90)

print("Fitting PCA...")
pca = PCA_AnomalyDetection(data, N_train, N_test)
pca.fit(6)

print("Fitting LSTM Autoencoder...")
lstm = LSTMAutoEncoder_AnomalyDetection(data, N_train, N_test, p, [
    LSTM(8, activation="relu", input_shape=(p,l*m), return_sequences=True),
    LSTM(4, activation="relu", return_sequences=False),
    RepeatVector(p),
    LSTM(4, activation="relu", return_sequences=True),
    LSTM(8, activation="relu", return_sequences=True),
    TimeDistributed(Dense(l*m))
])
lstm.fit(epochs=2)

print("Saving Models...")
with open("results/ltar_3dtpca.pkl", "wb") as f:
    pickle.dump(_3dtpca, f)
with open("results/ltar_lstm.pkl", "wb") as f:
    pickle.dump(lstm, f)
with open("results/ltar_pca.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("results/ltar_actual.pkl", "wb") as f:
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
    lstm.fit(epochs=2)
    end = time()
    lstm_times.append(end-start)

    start = time()
    pca.fit(6)
    end = time()
    pca_times.append(end-start)

times_df = pd.DataFrame()
times_df["3DTPCA"] = _3dtpca_times
times_df["LSTM"] = lstm_times
times_df["PCA"] = pca_times
print(times_df)
print(times_df.describe())
times_df.to_csv("results/ltar_times.csv", index=False)