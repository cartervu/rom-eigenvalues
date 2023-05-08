# From Wikipedia: https://en.wikipedia.org/wiki/Arnoldi_iteration
# Note that the Arnoldi algorithm is based on the modified Gram-Schmidt process

from math import ceil
import numpy as np

# from scipy.sparse import csr_matrix
# from prettytable import PrettyTable


def arnoldi_iteration(A, b=None, kmax=None, verbose=False, plot=False):
    # A is a csr_matrix from scipy.sparse
    # b is the random initial vector, also a csr_matrix
    # kmax is the maximum dimension of the krylov subspace, maximum number of vectors in our Krylov subspace
    dim = A.get_shape()[0]
    if kmax == None:
        kmax = ceil(dim / 10)  # if no kmax is explicitly defined, let kmax be the dimension of the matrix

    if b == None:
        b = np.random.randn(dim)
    
    print("kmax = ",kmax)

    eps = 1e-8
    h = np.zeros((kmax + 1, kmax))
    # h is the upper Hessenberg matrix we take the eigenvalues of. It has n+1 rows and n columns (or k+1, k, using notation of class)
    Q = np.zeros((dim, kmax + 1))
    # Q has same number of rows as A, but k+1 columns (one reserved for the initial guess).
    alleig = np.zeros((kmax,kmax),dtype=complex)
    # matrix to store vectors of eigenvalues at each iteration

    Q[:, 0] = b #/ np.linalg.norm(b, 2)  # 2-norm of input vector
    for k in range(1, kmax + 1):
        v = A.dot(Q[:, k - 1])  # this is the new candidate vector
        for j in range(k):  
            h[j, k - 1] = np.dot(Q[:, j].conj(), v)  
            # conjugate of all previous vectors dotted with the new vector. Don't know why we take the conjugate...
            v = v - h[j, k - 1] * Q[:, j]
            # subtract all components along previous vectors to orthogonalize by modified Gram-Schmidt
        h[k, k - 1] = np.linalg.norm(v, 2)  # the next value on the subdiagonal of h is the norm of the vector
        
        if plot == True: 
            #print(h)
            w = np.linalg.eig(h[0:k,0:k])[0]
            #print(w)
            alleig[0:k,k-1] = w # Store all eigenvalues as the columns of a matrix
        
        if h[k, k - 1] > eps: # if the new value is sufficiently large, we add it to our vector of Q, otherwise we finish
            Q[:, k] = v / h[k, k - 1]
        else:
            return Q, h, k, alleig
        
        if verbose:
            if k == 1 or k % 10 == 0:
                print("iteration number: ", k)
                print("hvalue: ", h[k, k - 1])
    print("dim(h): ", np.shape(h)[0], "x", np.shape(h)[1])
    return Q, h, kmax,alleig

    # would like to adaptively choose k -- and we do, if the k required is less than the kmax chosen, we converge to the tolerance before we reach the maximum kmax


####### TEST CSR_MATRIX
# if __name__ == "__main__":
#    testA = np.array([[4,7,6],[1,2,5],[9,3,8]])
#    x = PrettyTable()
#    x.add_rows(csr_matrix(testA))
# This errors out because the "length of a sparse matrix is ambiguous"
