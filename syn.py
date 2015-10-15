import numpy as np
import scipy as sp
from scipy import stats

#todo move to ./data

def xs(**kwargs):
    return np.linspace(0,1,1e3,dtype='float32')

def sinewave(xs,**kwargs):
    nw=kwargs.setdefault('nw',1) #num waves
    return np.sin(xs*nw*2*np.pi)

def const(xs,**kwargs):
    c=kwargs.setdefault('c',0)
    cs=np.empty(len(xs));cs.fill(c)
    return cs

def blip(xs,**kwargs):
    kwargs.setdefault('scale',.01)
    kwargs.setdefault('loc',.8)
    d=sp.stats.norm.pdf(xs,**kwargs)
    return d/max(d)

def pulse(xs,**kwargs):
    w= sinewave(xs,nw=100)\
       +blip(xs)*sinewave(xs,nw=300)
    return w


def pulsegen(**kwargs):
    return np.array(pulse(xs()),dtype='float32')

# #generate an even number of waves in the re
# nts=np.empty((100,len(xs)))
# for i in xrange(len(nts)):
#     nts[i]=sinewave(xs,nw=(i+1)*2)
# nts[-1]=const(xs,c=1)

def cyclespike(**kwargs):
    e=kwargs.setdefault('every',20)
    u=[0]*e
    u[-1]=1
    u=u*40
    u=np.array(u,dtype='float32')
    al=.8
    ali=int(len(u)*al)
    u[ali:ali+5]=.5
    return u
