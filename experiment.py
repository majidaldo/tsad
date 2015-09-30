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

#generate an even number of waves in the re
nts=np.empty((100,len(xs)))
for i in xrange(len(nts)):
    nts[i]=sinewave(xs,nw=(i+1)*2)
nts[-1]=const(xs,c=1)

nc=int(len(nts)*.5)
#pd=decomp.PCA(n_components=nc)
pd=decomp.FastICA(max_iter=int(3e3),n_components=nc)

pdc=pd.fit(nts)
#sigspc=pdc.components_

#find which u/s scheme gives least number of components
x=np.array([sinewave(xs,nw=4)*0+0*sinewave(xs,nw=7)+-10])

#recontruct signal
recon=pd.inverse_transform(pd.transform(x))

