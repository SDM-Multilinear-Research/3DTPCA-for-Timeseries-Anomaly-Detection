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
from sklearn.preprocessing import StandardScaler

import pickle

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

fast_size = 50

print("Performing the fast t-SVD...")
train = scaled[:N_train]
trans = np.fft.fft(train, axis=0)
trans = trans[:fast_size]
inv_trans = np.fft.ifft(trans, axis=0)
scaled = np.append(inv_trans, scaled[N_train:], axis=0)
N_train = fast_size

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

tprs_3dtpca, fprs_3dtpca, auc_3dtpca = get_roc_curve_values(_3dtpca, actual)
tprs_lstm, fprs_lstm, auc_lstm = get_roc_curve_values(lstm, actual)
tprs_pca, fprs_pca, auc_pca = get_roc_curve_values(pca, actual)

print("3DTPCA AUC =", auc_3dtpca)
print("LSTM AUC =", auc_lstm)
print("PCA AUC =", auc_pca)