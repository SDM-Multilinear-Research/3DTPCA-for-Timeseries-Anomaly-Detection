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

l, m = 30, 20
p = 5
N_train = 6460
N_test = 6782
print(f"N_train: {N_train}")
print(f"N_test: {N_test}")

train = pickle.load(open("data/UCSD/train.pkl", "rb"))
test = pickle.load(open("data/UCSD/test.pkl", "rb"))
actual = pickle.load(open("data/UCSD/labels.pkl", "rb"))

for i in range(N_train):
    for j in range(p):
        train[p-j-1] -= train[p-j-1]

for i in range(N_test):
    for j in range(p):
        test[p-j-1] -= test[p-j-1]

train = np.delete(train, p-1, 2)
test = np.delete(test, p-1, 2)
p -= 1 

print(train.shape)
print(test.shape)
    

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
plt.title("ROC Curve for UCSD")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(["Baseline", 
    f"3DTPCA : AUC = {np.round(auc_3dtpca,4)}",
    f"LSTM AutoEncoder : AUC = {np.round(auc_lstm,4)}",
])
plt.savefig("UCSD_ROC.png")
plt.close()