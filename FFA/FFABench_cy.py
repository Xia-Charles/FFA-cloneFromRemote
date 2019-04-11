import numpy as np
from numpy import ma
import FFA_cy as FFA

def FFABench():
   def seg(X,P0):
       XW = FFA.XWrap2(X,P0,pow2=True)
       M   = XW.shape[0]  # number of rows
       
       idCol = np.arange(P0,dtype=int)   # id of each column
       idRow = np.arange(M,dtype=int)   # id of each row
       P  = P0 + idRow.astype(float) / (M - 1)
   
       XW.fill_value=0
       data = XW.filled()
       mask = (~XW.mask).astype(int)
   
       sumF   = FFA.FFA(data) # Sum of array elements folded on P0, P0 + i/(1-M)
       countF = FFA.FFA(mask) # Number of valid data points
       meanF  = sumF/countF
   
       names = ['mean','count','s2n','P']
       dtype = zip(names,[float]*len(names) )
       rep   = np.empty(M,dtype=dtype)
       
       # Take maximum epoch
       idColMa      = meanF.argmax(axis=1)
       rep['mean']  = meanF[idRow,idColMa]
       rep['count'] = countF[idRow,idColMa]
       rep['s2n']   = rep['mean']*np.sqrt(rep['count'])
       rep['P']     = P
   
       return rep
   X = np.load('pulse_train_data.npy')
   Xmask = np.load('pulse_train_mask.npy')
   X = ma.masked_array(X,Xmask,fill_value=0)
   
   X = X[:10000] # Modify this to change execution time.
   
   Pmin,Pmax = 250,2500
   PGrid = np.arange(Pmin,Pmax)
   
   ''' Just learn git pull command line, then should delete '''
   for b in range(len(Bins)):
        binInds = range(offset-1,offset+Bins[b])
        T[binInds] = 4*N*noise_var[b] + threshold
        Bin_sizes[binInds] = Bins[b]
        Bin_offset[binInds] = offset-1
        offset = offset + Bins[b]
    offsets = np.ones(len(Bins),dtype=np.int)   
   @cython.boundscheck(False)
@cython.wraparound(False)
def cmaxDelTt0( cnp.ndarray[double, ndim=2,mode='c'] XsumP,
                cnp.ndarray[double, ndim=2,mode='c'] XXsumP,
                cnp.ndarray[double, ndim=2,mode='c'] XcntP,
                int P,
                cnp.ndarray[int, ndim=1,mode='c'] DelTarr,
                int nDelT):
                

    cdef int M = XsumP.shape[0]
    cdef int j                                     ## delete this paragraph
   
   func = lambda P0: seg(X,P0)
   rep = map(func,PGrid)
   rep = np.hstack(rep)
   return rep

