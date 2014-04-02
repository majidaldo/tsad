import numpy as np
import scipy as sp
from sklearn import decomposition as decomp

xs=np.linspace(0,1,10e3)
ts=np.sin(xs*100*2*np.pi)+np.cos(xs*50*2*np.pi)

nts=np.reshape(ts,(10,int(1e3)))
ntsf=np.empty((10,len(ts)/10))#noisy ts
for atsi in xrange(len(nts)):
    cts=ts[atsi*1000:(atsi+1)*1000]
    ntsf[atsi]=cts#\
    #+.05*sp.std(cts)*sp.random.randn(len(cts))
    

#if the same sample then should reproduce
    
pd=decomp.PCA(n_components=5)
#pd=decomp.RandomizedPCA(n_components=3)#'mle')
#pd=decomp.FastICA(n_components=3)

pdc=pd.fit(nts)


