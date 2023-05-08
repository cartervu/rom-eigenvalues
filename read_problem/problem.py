from scipy.sparse import csr_matrix
from scipy.io import mmread


class eigmatrix:
    def __init__(self, matrixname):
        filename = 'read_problem/' + matrixname + '.mtx'
        M = mmread(filename)
        self.values = csr_matrix(M)
