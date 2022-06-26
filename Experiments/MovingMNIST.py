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

from time import time

import pickle

N = 5000
l, m = 50, 50
data = np.zeros((N,l,m))
for i in range(N):
    data[i] = np.loadtxt(f"data/MovingMNIST/frame{i}.txt")

p = 10
N = len(data)
N_test = 1000
N_train = N - N_test
actual = np.zeros((N_test,),dtype=int)
actual[500:800] = 1
print(f"N: {N}")
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

def denoise(fold):
    fold[fold < 125] = 0
    fold[fold >= 125] = 255
    return fold

print("Fitting 3DTPCA..")
_3dtpca = _3DTPCA_AnomalyDetection(data, N_train, N_test, p, economy_mode=True, debug=True)
_3dtpca.fit(0.90, fast_size=100, post_process=denoise)

print("Fitting LSTM Autoencoder...")
lstm = LSTMAutoEncoder_AnomalyDetection(data, N_train, N_test, p, [
    LSTM(1024, activation="relu", input_shape=(p,l*m), return_sequences=True),
    LSTM(256, activation="relu", return_sequences=True),
    LSTM(64, activation="relu", return_sequences=False),
    RepeatVector(p),
    LSTM(64, activation="relu", return_sequences=True),
    LSTM(256, activation="relu", return_sequences=True),
    LSTM(1024, activation="relu", return_sequences=True),
    TimeDistributed(Dense(l*m))
])
lstm.fit(epochs=5, post_process=denoise)

print("Fitting PCA...")
pca = PCA_AnomalyDetection(data, N_train, N_test)
pca.fit(25)

print("Saving Models...")
with open("results/MovingMNIST_3dtpca.pkl", "wb") as f:
    pickle.dump(_3dtpca, f)
with open("results/MovingMNIST_lstm.pkl", "wb") as f:
    pickle.dump(lstm, f)
with open("results/MovingMNIST_pca.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("results/MovingMNIST_actual.pkl", "wb") as f:
    pickle.dump(actual, f)

# print("Testing Speed")
# _3dtpca_times = [] 
# lstm_times = [] 
# pca_times = [] 
# for i in range(20):

#     print("Running trial", i)

#     start = time()
#     _3dtpca.fit(0.90, fast_size=100, post_process=denoise)
#     end = time()
#     _3dtpca_times.append(end-start)

#     start = time()
#     lstm.fit(epochs=5, post_process=denoise)
#     end = time()
#     lstm_times.append(end-start)

#     start = time()
#     pca.fit(25)
#     end = time()
#     pca_times.append(end-start)

# times_df = pd.DataFrame()
# times_df["3DTPCA"] = _3dtpca_times
# times_df["LSTM"] = lstm_times
# times_df["PCA"] = pca_times
# print(times_df)
# print(times_df.describe())
# times_df.to_csv("results/MovingMNIST_times.csv", index=False)