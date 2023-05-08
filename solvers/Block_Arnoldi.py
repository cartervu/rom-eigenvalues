import numpy as np
# This is the block arnoldi method with Block Modified Gram-Schmidt Orthogonalization (to reduce numerical errors vs the standard Gram-Schmidt)
# From page 198, Algorithm 6.22 from Yousef Saad's "Iterative Methods for Sparse Linear Systems"

def Block_Arnoldi_Iteration(A,p,V=None,kmax=None,verbose=False,plot=False):
    dim = A.get_shape()[0]
    if kmax == None:
        kmax = np.ceil(dim / 10)
    if V == None:
        temprand = np.random.rand(p,p)
        Q = np.linalg.qr(temprand)[0]
        V = np.zeros((dim,p))
        V[0:p,0:p] = Q
        print("Orthogonal?")
        print(np.matmul(Q,Q.conj().T))
        #V[0:p,0:p] = temprand # Try nonorthongal V to see what happens...
        # use QR decomposition to calculate an nxp unitary matrix. This is very expensive, and could use a much less expensive initialization.
        # At this point, we might as well perform the QR decomposition of the whole matrix and use that to directly compute the eigenvalues
        # Perhaps I could start from a pxp matrix, orthogonalize that, and then add a bunch of zeros? Would this be a unitary matrix? Yes.
        # This is much better.

    print("kmax = ", kmax)
    print("p = ", p)
        
    h = np.zeros(((kmax+1)*p,kmax*p))
    Q = np.zeros((dim, (kmax+1)*p))
    alleig = np.zeros((kmax*p,kmax*p),dtype=complex)

    for col in range(p):
        colvals = V[:,col]
        Q[:, col] = colvals / np.linalg.norm(colvals, 2)
        # Normalize each vector in v and add it to Q 
    
    for k in range(1,kmax+1):
        V = A.dot(Q[:,(k-1)*p:k*p]) #This is the new candidate block -- check that the appropriate block of Q is being selected...
        #print(V)
        for j in range(k):
            h[j*p:(j+1)*p,(k-1)*p:k*p] = np.matmul(Q[:,j*p:(j+1)*p].conj().T,V) # Take the conjugate transpose
            V = V - np.dot(Q[:,j*p:(j+1)*p],h[j*p:(j+1)*p,(k-1)*p:k*p]) # Vj+1(orthogonal) = Vj+1 - all previous products of Vj and H
        # Need to figure out how to assign the elements on the lower diagonal, if they are not being assigned -- how do you do this in a block method?
            #for col in np.arange(1,p+1):
                #diagindex = (k-1)*p+col
                #print(diagindex)
                #print(h)
                #h[diagindex,diagindex-1] =  2#np.linalg.norm(V[:,col], 2)
        
        if plot == True: 
            #print(h)
            w = np.linalg.eig(h[0:k*p,0:k*p])[0]
            #print(w)
            alleig[0:k*p,k-1] = w # Store all eigenvalues as the columns of a matrix
        
        #for col in np.arange(1,p+1):
        #    diagindex = (k-1)*p+col 
        #    Q[:, diagindex] = V[:,col-1] / h[diagindex, diagindex-1] # Is this normalization right??
        # add the new block of vectors to the orthnormal Q block. Have to normalize though...
        
        # Compute QR decomposition of the newly orthogonalized V and use Q as the next V, add the R to H as the subdiagonal H block
        V, R = np.linalg.qr(V)
        Q[:,k*p:(k+1)*p] = V # Note that I don't normalize these vectors at any point!! Should I??? It doesn't say to do it in the algorithm, so I don't.
        h[k*p:(k+1)*p,(k-1)*p:k*p] = R

        if verbose:
            #if k == 1 or k % 10 == 0:  # we don't use the if statement since this will probably only run for 5 or so iterations anyways
                print("iterations complete: ", k)
                #print("hvalue: ", h[k, k - 1])
    print("dim(h): ", np.shape(h)[0], "x", np.shape(h)[1])
    return Q, h, kmax,alleig



