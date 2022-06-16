import numpy as np
from tensorflow import keras
from keras.models import Sequential

from _3DTPCA import _3DTPCA
from ltar import LTAR
from sklearn.metrics import confusion_matrix, roc_curve, auc

def get_roc_curve_values(model, actual):
    fpr, tpr, thresholds = roc_curve(actual, model.test_err)
    roc_auc = auc(fpr, tpr)
    return tpr, fpr, roc_auc

class LTAR_AnomalyDetection:
    def __init__(self, data, N_train, N_test) -> None:
        self.train = data[:N_train]
        self.test = data[N_train:]
        self.model = LTAR(self.train)
        self.N_test = N_test
    
    def fit(self, p, transformation="dct", post_process = None):
        self.model.fit(p, transformation)
        self.test_forecast = self.model.single_step_forecast(self.N_test, self.test)
        if post_process is not None:
            self.test_forecast = post_process(self.test_forecast)
        self.test_err = self.err(self.test, self.test_forecast)

    def err(self, true, recon):
        N = true.shape[0]
        err_arr = np.zeros((N,))
        for i in range(N):
            err_arr[i] = np.linalg.norm(true[i,:,:] - recon[i,:,:])
        return err_arr

    def predict(self, threshold_percent):
        threshold = np.percentile(self.train_err, threshold_percent)
        pred = np.zeros((len(self.test_err),),dtype=int)
        pred[self.test_err > threshold] = 1
        return pred

class LSTMAutoEncoder_AnomalyDetection:
    def __init__(self, data, N_train, N_test, p, layers, test=None, need_fold=True) -> None:
        self.model = Sequential(layers)
        self.model.compile(optimizer="adam",loss="mse")
        if need_fold:
            (N, l, m) = data.shape
            self.train_fold, self.test_fold = self.fold(data, N_train, N_test, l, m, p)
        else:
            self.train_fold = data
            self.test_fold = test

    def fold(self, data, N_train, N_test, l, m, p):

        train = data[:N_train]
        test = data[N_train-p:]

        train_fold = np.zeros((N_train-p,p,l*m))
        for i in range(p,N_train):
            curr_tensor = np.zeros((p,l*m))
            for j in range(p):
                curr_tensor[p-j-1,:] = train[i-j,:,:].reshape((l*m,))
            train_fold[i-p,:,:] = curr_tensor

        test_fold = np.zeros((N_test,p,l*m))
        for i in range(p, N_test+p):
            curr_tensor = np.zeros((p,l*m))
            for j in range(p):
                curr_tensor[p-j-1,:] = test[i-j,:,:].reshape((l*m,))
            test_fold[i-p,:,:] = curr_tensor
        return train_fold, test_fold

    def err(self, true, recon):
        N = true.shape[0]
        err_arr = np.zeros((N,))
        for i in range(N):
            err_arr[i] = np.linalg.norm(true[i,:,:] - recon[i,:,:])
        return err_arr

    def fit(self, epochs, post_process = None):
        self.model.fit(self.train_fold, self.train_fold, epochs=epochs)
        self.train_recon = self.model.predict(self.train_fold)
        self.test_recon = self.model.predict(self.test_fold)
        if post_process is not None:
            self.train_recon = post_process(self.train_recon)
            self.test_recon = post_process(self.test_recon)
        self.train_err = self.err(self.train_fold, self.train_recon)
        self.test_err = self.err(self.test_fold, self.test_recon)

    def predict(self, threshold_percent):
        threshold = np.percentile(self.train_err, threshold_percent)
        pred = np.zeros((len(self.test_err),),dtype=int)
        pred[self.test_err > threshold] = 1
        return pred

class _3DTPCA_AnomalyDetection:
    def __init__(self, data, N_train, N_test, p, test=False, need_fold=True, economy_mode=False, debug=False) -> None:
        self.economy_mode = economy_mode
        self.debug = debug
        self.model = _3DTPCA(economy_mode, debug)
        if need_fold:
            (N, l, m) = data.shape
            self.train_fold, self.test_fold = self.fold(data, N_train, N_test, l, m, p)
        else:
            self.train_fold = data
            self.test_fold = test

    def err(self, true, recon):
        N = true.shape[-1]
        err_arr = np.zeros((N,))
        for i in range(N):
            err_arr[i] = np.linalg.norm(true[:,:,:,i] - recon[:,:,:,i])
        return err_arr

    # Takes a N x l x m tensor and folds it into a l x `    m x N x p tensor
    def fold(self, data, N_train, N_test, l, m, p):

        train = data[:N_train]
        test = data[N_train-p:]

        train_fold = np.zeros((l, m, p, N_train-p))
        for i in range(p,N_train):
            curr_tensor = np.zeros((l, m, p))
            for j in range(p):
                curr_tensor[:,:,p-j-1] = train[i-j,:,:]
            train_fold[:,:,:,i-p] = curr_tensor

        test_fold = np.zeros((l, l, p, N_test))
        for i in range(p, N_test+p):
            curr_tensor = np.zeros((l, l, p))
            for j in range(p):
                curr_tensor[:,:,p-j-1] = test[i-j,:,:]
            test_fold[:,:,:,i-p] = curr_tensor
        
        return train_fold, test_fold

    def fit(self, energy_ratio, post_process = None):
        self.model.fit(self.train_fold, energy_ratio=energy_ratio)
        self.train_recon = self.model.inverse_transform(self.model.transform(self.train_fold))
        self.test_recon = self.model.inverse_transform(self.model.transform(self.test_fold))
        if post_process is not None:
            self.train_recon = post_process(self.train_recon)
            self.test_recon = post_process(self.test_recon)
        self.train_err = self.err(self.train_fold, self.train_recon)
        self.test_err = self.err(self.test_fold, self.test_recon)

    def predict(self, threshold_percent):
        threshold = np.percentile(self.train_err, threshold_percent)
        pred = np.zeros((len(self.test_err),),dtype=int)
        pred[self.test_err > threshold] = 1
        return pred
