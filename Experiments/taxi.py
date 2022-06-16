import sys
sys.path.insert(0, '/home/jcates/repos/multi_linear_research/jackson/Libraries')
from AnomalyDetection import _3DTPCA_AnomalyDetection, LSTMAutoEncoder_AnomalyDetection, LTAR_AnomalyDetection, get_roc_curve_values

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from sklearn.preprocessing import StandardScaler

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
_3dtpca = _3DTPCA_AnomalyDetection(scaled, N_train, N_test, p, economy_mode=True, debug=False)
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

print("Fitting LTAR...")
ltar = LTAR_AnomalyDetection(data, N_train, N_test)
ltar.fit(p=5)

print("Generating ROC Curves...")
tprs_3dtpca, fprs_3dtpca, auc_3dtpca = get_roc_curve_values(_3dtpca, actual)
tprs_lstm, fprs_lstm, auc_lstm = get_roc_curve_values(lstm, actual)
tprs_ltar, fprs_ltar, auc_ltar = get_roc_curve_values(ltar, actual)
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), "r--")
plt.plot(fprs_3dtpca,tprs_3dtpca)
plt.plot(fprs_lstm,tprs_lstm)
plt.plot(fprs_ltar,tprs_ltar)
plt.title("ROC Curve for Taxi")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(["Baseline", 
    f"3DTPCA : AUC = {np.round(auc_3dtpca,4)}",
    f"LSTM AutoEncoder : AUC = {np.round(auc_lstm,4)}",
    f"LTAR : AUC = {np.round(auc_ltar,4)}"
])
plt.savefig("Taxi_ROC.png")
plt.close()

print("Generating Error Plots...")
norms = []
for i in range(N_train, N):
    norms.append(np.linalg.norm(data[i]))
fig, axs = plt.subplots(4, 1, figsize=(14,7))
axs[0].plot(norms)
axs[1].plot(_3dtpca.test_err)
axs[2].plot(lstm.test_err)
axs[3].plot(ltar.test_err)
axs[0].set_title("Norm of Taxi")
axs[1].set_title("Error for 3DTPCA")
axs[2].set_title("Error for LSTM")
axs[3].set_title("Error for LTAR")
plt.savefig("Taxi_ERR.png")
plt.close()