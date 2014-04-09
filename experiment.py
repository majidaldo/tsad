import numpy as np
import scipy as sp
from sklearn import decomposition as decomp

xs=np.linspace(0,1,10e3)

def sinewave(xs,**kwargs):
    nw=kwargs.setdefault('nw',100) #num waves
    return np.sin(xs*nw*2*np.pi)

def const(xs,**kwargs):
    c=kwargs.setdefault('c',0)
    cs=np.empty(len(xs));cs.fill(c)
    return cs

nts=np.empty((1000,len(xs)))
for i in xrange(len(nts)):
    nts[i]=sinewave(xs,nw=(i+1)*2)


nc=int(len(nts)*.5)
#pd=decomp.PCA(n_components=nc)
#pd=decomp.KernelPCA(n_components=nc)
#pd=decomp.RandomizedPCA(n_components=3)#'mle')
pd=decomp.FastICA(max_iter=int(3e3),n_components=nc)
#pd=decomp.SparsePCA(n_components=nc,n_jobs=4) #NOT sparse coding! just alot of zeros..too slow
pdc=pd.fit(nts)
#sigspc=pdc.components_

#find which u/s scheme gives least number of components
x=np.array([sinewave(xs,nw=4)+sinewave(xs,nw=6)])
#x=np.array([const(xs,c=3)]) #zero is trivial though

#sc=decomp.sparse_encode( x,  sigspc )

#scc=sc.transform( x  )
#sc=decomp.SparseCoder(np.array(  [[1.,0],[333,0]]   ))
#scc=sc.transform( nts  )

#need to be careful about orthogonal. it the folowing ok?
#reconstruct signal

#recon=np.dot(sc,sigspc) # assumes mean is zero?
#can do pd.transform
recon=pd.inverse_transform(pd.transform(nts))

