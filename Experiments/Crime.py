from copyreg import pickle
import sys
sys.path.insert(0, '/home/jcates/repos/multi_linear_research/jackson/Libraries')
from AnomalyDetection import _3DTPCA_AnomalyDetection, LSTMAutoEncoder_AnomalyDetection, LTAR_AnomalyDetection, get_roc_curve_values

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

l, m = 64, 64
p = 4
N_train = 50000
N_test = 6321
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

train = pickle.load(open("data/Crime/train.pkl", "rb"))[:,:,-p:,:N_train]
test = pickle.load(open("data/Crime/test.pkl", "rb"))[:,:,-p:,:N_test]
actual = pickle.load(open("data/Crime/labels.pkl", "rb"))[:N_test]

print(train.shape, test.shape, actual.shape)

print("Fitting 3DTPCA..")
_3dtpca = _3DTPCA_AnomalyDetection(train, N_train, N_test, p, test=test, need_fold=False, economy_mode=True, debug=True)
_3dtpca.fit(0.95)

print("Fitting LSTM Autoencoder...")
lstm_train = np.transpose(train, (3, 2, 0, 1)).reshape((N_train, p, l*m))
lstm_test = np.transpose(test, (3, 2, 0, 1)).reshape((N_test, p, l*m))
lstm = LSTMAutoEncoder_AnomalyDetection(lstm_train, N_train, N_test, p, [
    LSTM(1024, activation="relu", input_shape=(p,l*m), return_sequences=True),
    LSTM(256, activation="relu", return_sequences=True),
    LSTM(64, activation="relu", return_sequences=False),
    RepeatVector(p),
    LSTM(64, activation="relu", return_sequences=True),
    LSTM(256, activation="relu", return_sequences=True),
    LSTM(1024, activation="relu", return_sequences=True),
    TimeDistributed(Dense(l*m))
], test=lstm_test, need_fold=False,)
lstm.fit(epochs=5)

print("Generating ROC Curves...")
tprs_3dtpca, fprs_3dtpca, auc_3dtpca = get_roc_curve_values(_3dtpca, actual)
tprs_lstm, fprs_lstm, auc_lstm = get_roc_curve_values(lstm, actual)
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), "r--")
plt.plot(fprs_3dtpca,tprs_3dtpca)
plt.plot(fprs_lstm,tprs_lstm)
plt.title("ROC Curve for Crime")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(["Baseline", 
    f"3DTPCA : AUC = {np.round(auc_3dtpca,4)}",
    f"LSTM AutoEncoder : AUC = {np.round(auc_lstm,4)}",
])
plt.savefig("Crime_ROC.png")
plt.close()