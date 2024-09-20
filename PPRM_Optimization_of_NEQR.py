import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


# In[2]:


# Dictionary of image filenames
image_files = {
    'Aerial': 'sipi.usc.edu_misc/misc/5.1.10.tiff',
    'Female': 'sipi.usc.edu_misc/misc/4.1.04.tiff',
    'Moon': 'sipi.usc.edu_misc/misc/5.1.09.tiff',
    'Clock': 'sipi.usc.edu_misc/misc/5.1.12.tiff',
    'Airplane': 'sipi.usc.edu_misc/misc/5.1.11.tiff',
    'Jelly': 'sipi.usc.edu_misc/misc/4.1.07.tiff'
}

# Corresponding display titles
display_titles = {
    'Aerial': 'Aerial 5.1.10',
    'Female': 'Female 4.1.04',
    'Moon': 'Moon surface 5.1.09',
    'Clock': 'Clock 5.1.12',
    'Airplane': 'Airplane 5.1.11',
    'Jelly': 'Jelly beans 4.1.07'
}

# Load images into a dictionary
images = {name: np.array(Image.open(file).convert('L')).astype(np.uint8)
          for name, file in image_files.items()}



# # Two ways of MCNOT cost calculations

# In[3]:


def mcnot_qc(m):
    '''
    The QC of MCNOT without any ancillary qubits, described in Barenco et al.
    m = number of control qubits
    The QC of Toffoli and CNOT gates are 6 and 1, respectively.
    '''
    if m>2:
        return 3* (2**m) - 4
    elif m==2:
        return 6
    elif m==1:
        return 1

def mcnotR_qc(m):
    '''
    The QC of MCNOT-R with 2 ancillary qubits, described in Iranmanesh et al.
    m = number of control qubits
    The QC of Toffoli and CNOT gates are 6 and 1, respectively.
    '''
    if m>2:
        return 19*m - 32
    elif m==2:
        return 6
    elif m==1:
        return 1


# # QC without Optimization

# In[4]:


def QC_old(image, mcnot):
    '''
    image_binary is a matrix which has rows as many pixel positions and columns as many
    grayscale bits.
    A matrix in which each column indicates the "b" vector of the specified grayscale bit.
    Each row is in a descending order (7 to 0). 
    As an example for a row, 56 -> array([0, 0, 1, 1, 1, 0, 0, 0], dtype=uint8)
    
    Each column of "i" indicates which pixel positions should convert the 
    grayscale qubit of "i" from 0 to 1, meaning that each 1s in this column 
    indicates how many MCNOTs required for the grayscale qubit of "i".
    
    mcnot: mcnot_qc(m), mcnotR_qc(m)
    '''
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    # m = number of control qubits
    m = np.log2(dim1*dim2)
    image_binary = np.unpackbits(image, axis=1, bitorder='big').reshape(dim1*dim2, 8)

    return np.count_nonzero(image_binary) * mcnot(m)


# In[6]:



# # *************************************************
# ### PPRM Expressions


# In[11]:


class PPRM:
    
    def __init__(self, a):
        '''
        a: input vector, indicating the coefficients of f 
        '''
        self.a = a.astype(np.uint8)
        self.k = np.log2(self.a.shape[0]).astype(np.uint8)
        self.b = self.Binary_PPRM(self.a, self.k)
        self.X = self.X(self.k)

    
    def Binary_PPRM(self, a, k):
        blocksize = 1
        for i in range(k):
            mask = np.array([[1]*(2**i), [0]*(2**i)]*(2**(k-(i+1)))).reshape(2**k,1).astype(np.uint8)
            a_mask = a & mask; # masks the blocks;
            # Create a new array of the same shape filled with zeros
            temp = np.zeros_like(a_mask)
            # Copy elements from the original array to the new array starting from blocksize index
            # temp = temp SHR blocksize; //shift right
            temp[blocksize:] = a_mask[:-blocksize]
            # XOR between all blocks.
            a = a ^ temp; 
            blocksize *= 2
        return a


    def x_product(self, x_i, x_j):
        '''
        x_i: [i],  x_j: [j]
        x_product([i], [j]) = [i, j]   : x_i . x_j
        x_product(x_i, 1) = [i]        : x_i . 1 = x_i

        For example:
           rm_concat([2], 1) = [2]            : x2 . 1 = x2
           rm_concat([2], [0]) = [0, 2]       : x0 . x2 
           rm_concat([0, 1], [2]) = [0, 1, 2] : x0 . x1 . x2
       '''
        if x_i == 1:
            return x_j
        if x_j == 1:
            return x_i
        return list(np.concatenate((x_j, x_i)))
    

    def x_kron(self, list2, list1):
        ''' kronecker product of lists of product terms
            x_kron([1,[1]], [1,[0]]) = [1, [0], [1], [0, 1]]
        '''
        output = list1
        for i in range(len(list1)):
            output.append(self.x_product(list2[1], list1[i]))
        return output
    

    def X(self, k):
        # time0 = time.perf_counter_ns()
        '''
        X(1) = [1, [0]]               : 1 xor x0

        X(2) = [1, [0], [1], [0, 1]]  : 1 xor x0 xor x1 xor x0.x1
                
        X(3) = [1, [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]] 
        '''
        if k == 1:
            return [1, [0]]
        if k == 2:
            return self.x_kron([1, [1]], [1, [0]])

        result = [1, [0]]  # Start with the result for k = 1
        for i in range(2, k+1):
            result = self.x_kron([1, [i-1]], result)
            
        # time1= time.perf_counter_ns() - time0
        # print("time1:", time1/(10**9), " seconds")
        return result


    
    def pprm(self):
        ''' Returns a list indicating which product terms are required
        for the new PPRM expression.

        For example:

            b = array([[0],
                       [1],
                       [1],
                       [0],
                       [1],
                       [0],
                       [0],
                       [0]])

            pprm() = [[0], [1], [2]]  : x0 xor x1 xor x2
        '''
        result = []
        idx_ones_in_b = np.argwhere(self.b == 1)[:, 0].reshape(-1)
        for idx in idx_ones_in_b:
            result.append(self.X[idx])
        return result


# # Calculate PPRM QC

# In[13]:


def QC(image, mcnot):
    '''mcnot: mcnot_qc or mcnotR_qc'''
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    qc_sum = 0
    image_binary = np.unpackbits(image, axis=1, bitorder='big').reshape(dim1*dim2, 8).astype(np.uint8)
    
    for i in range(8):
        a = image_binary[:, i].reshape(-1, 1)
        pprm_instance = PPRM(a)
        pprm = pprm_instance.pprm()
    
        for product_term in pprm:
            # Skip the QC of X Gates
            if product_term == 1:
                continue
            else:
                # len(product_term) = the number of control qubits of the MCNOT
                qc_sum = qc_sum + mcnot(len(product_term))
    return qc_sum

    

