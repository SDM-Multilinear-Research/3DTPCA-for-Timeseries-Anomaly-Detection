import numpy as np
import cupy as cp

class _3DTPCA:
    def __init__(self, economy_mode=False, debug=False): # TODO allow different transforms
        self.U = []
        self.debug = debug
        self.economy_mode = economy_mode
        self.transpose_order = [(0,2), (1,2)]

    def _find_energy_ratio(self, S, A, energy_ratio):
        if self.debug:
            print("--Finding energy recovery ratio...")
        sum = 0
        k = 0
        while sum / np.linalg.norm(A)**2 < energy_ratio:
            sum += np.linalg.norm(S[:,:,k,k])**2
            k += 1
        return k

    def _3DTPCA(self, A, i, j, k, energy_ratio, fast_size):

        # Compute the SVD
        self.SVD = []
        self.U = []
        keep_order = [i, j]

        # Checks if we are doing the fast t-SVD
        if fast_size is not None:
            if self.debug:
                print("Completing Fast t-SVD")
            if fast_size < 0:
                raise ValueError("Invalid Fast Size. Must be greater than 0.")
            trans = np.fft.fft(A, axis=3) # Do the FFT down the samples
            A = trans[:,:,:,:fast_size] # Only keep the low frequency information
            A = np.fft.ifft(A, axis=3) # Inverse transform

        if self.debug:
            print("Completing Mode 1")

        # Compute the tSVD
        U, S, V = self.tSVD(A)

        # Check if we are doing energy ratio mode
        if energy_ratio is not None:
            k = self._find_energy_ratio(S, A, energy_ratio)

        # Keep the principle components
        U_k = U[:,:,:,:k]
        self.U.append(U_k)

        # Project onto the new subspace 
        Y = self.tprod(self.ttrans(U_k), A)

        for step, axis in enumerate(self.transpose_order):

            if self.debug:
                print(f"Completing Mode {step+2}")

            # Tensor Rotate
            Y = self.trot(Y, axis)

            # Compute the tSVD
            U, S, V = self.tSVD(Y)

            # Check if we are doing energy ratio mode
            if energy_ratio is not None:
                keep_order[step] = self._find_energy_ratio(S, Y, energy_ratio)

            # Keep the principle components
            U_k = U[:,:,:,:keep_order[step]]
            self.U.append(U_k)

            # Project onto the new subspace 
            Y = self.tprod(self.ttrans(U_k), Y)

        if self.debug:
            print("Final Dimension is ", Y.shape)
        

    def tSVD(self, A):
        if self.debug:
            print("--Calculating t-SVD...")
        n0,n1,n2,n3 = A.shape
        if self.debug:
            print("----Calculating FFT...")
        Ahat = np.fft.fft(A,axis=1)
        Ahat = np.fft.fft(Ahat,axis=0)
        U = np.zeros((n0,n1,n2,n2),dtype="complex")
        S = np.zeros((n0,n1,n2,n3),dtype="complex")
        # V = np.zeros((n0,n1,n3,n3),dtype="complex")
        if self.debug:
            print("----Calculating SVD...")
        for i in range(n0):
            for j in range(n1):
                u,s,v = np.linalg.svd(Ahat[i,j,:,:],full_matrices=(not self.economy_mode))
                np.fill_diagonal(S[i,j, :, :], s)
                U[i,j,:,:] = u
                #V[i,j,:,:] = (np.conj(v)).T
        if self.debug:
            print("----Calculating Inverse FFT...")
        U = np.fft.ifft(U,axis=0)
        U = np.fft.ifft(U,axis=1)
        S = np.fft.ifft(S, axis=0)
        S = np.fft.ifft(S, axis=1)
        # V1 = np.fft.ifft(V, axis=0)
        # V2 = np.fft.ifft(V1, axis=1)
        return U,S, None #,V2

    def tprod(self, A, B):
        if self.debug:
            print("--Calculating tprod...")
        n0,n1,n2,n3 = A.shape
        m0,m1,m2,m3 = B.shape
        if n0 != m0 and n1!=m1 and n3!=m2:
            print('warning, dimensions are not acceptable')
            return
        Ahat = np.fft.fft(A, axis=1)
        Ahat = np.fft.fft(Ahat, axis=0)
        Bhat = np.fft.fft(B, axis=1)
        Bhat = np.fft.fft(Bhat, axis=0)
        C = np.zeros((n0,n1,n2,m3),dtype="complex")
        for i in range(n0):
            for j in range(n1):
                C[i,j,:,:] = Ahat[i,j,:,:]@Bhat[i,j,:,:]
        C = np.fft.ifft(C,axis=0)
        C = np.fft.ifft(C,axis=1)
        return C.real

    def trot(self, A, axiss):
        if self.debug:
            print("--Performing tensor rotation...")
        newShape = np.arange(4)
        axis0, axis1 = axiss
        newShape[axis0], newShape[axis1] = newShape[axis1], newShape[axis0]
        return np.transpose(A, newShape)

    def ttrans(self, A):
        if self.debug:
            print("--Performing tensor transpose...")
        n0, n1, n2, n3 = A.shape
        Ahat = np.fft.fft(A, axis=1)
        Ahat = np.fft.fft(Ahat, axis=0)
        B = np.zeros((n0,n1,n3,n2),dtype="complex")
        for i in range(n0):
            for j in range(n1):
                B[i,j,:,:] = np.transpose(np.conj(Ahat[i,j,:,:]))
        B1 = np.fft.ifft(B,axis=0)
        B2 = np.fft.ifft(B1,axis=1)
        return B2.real

    def fit(self, A, i=None, j=None, k=None, energy_ratio=None, fast_size=None):

        (p, l, m, n_p) = A.shape # This tensor is p x l x m x n_p

        if (i is not None or j is not None or k is not None) and energy_ratio is not None:
            print("WARNING: energy_ratio mode will overrite ijk")

        if i == None:
            i = p
        if j == None:
            j = l
        if k == None:
            k = m

        # Does PCA
        self._3DTPCA(A, i, j, k, energy_ratio, fast_size)

    def transform(self, A):
        
        if self.debug:
            print("Completing Mode 1")

        # Y = U^t * A
        Y = self.tprod(self.ttrans(self.U[0]), A)

        # Project into PCA space
        for step, axis in enumerate(self.transpose_order):

            if self.debug:
                print(f"Completing Mode {step+2}")

            # Tensor Rotate
            Y = self.trot(Y, axis)

            # Y = U^t * A
            Y = self.tprod(self.ttrans(self.U[step+1]), Y)

        return Y

    def inverse_transform(self, A):
        
        if self.debug:
            print("Completing Mode 1")

        # Y = U * A 
        # Y = U^t * A
        Y = self.tprod(self.U[2], A)

        # Project into PCA space
        for step in range(2):

            if self.debug:
                print(f"Completing Mode {step+2}")

            # Tensor Rotate
            Y = self.trot(Y, self.transpose_order[1-step])

            # Y = U^t * A
            Y = self.tprod(self.U[1-step], Y)

        return Y

# model = _3DTPCA()
# np.random.seed(99)
# A = np.random.normal(size=(2, 3, 4, 5))
# model.fit(A)
# Atransform = model.transform(A)
# Ahat = model.inverse_transform(Atransform)
# print(A.shape)
# print(Atransform.shape)
# print(Ahat.shape)
