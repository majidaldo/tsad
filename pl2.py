import numpy as np
import scipy as sp

xs=np.linspace(0,1,10e3)

def sinewave(xs,**kwargs):
    nw=kwargs.setdefault('nw',100) #num waves
    return np.sin(xs*nw*2*np.pi)

def const(xs,**kwargs):
    c=kwargs.setdefault('c',0)
    cs=np.empty(len(xs));cs.fill(c)
    return cs

#generate an even number of waves in the re
nts=np.empty((100,len(xs)),dtype='float32')
for i in xrange(len(nts)):
    nts[i]=sinewave(xs,nw=(i+1)*2)
nts[-1]=const(xs,c=1000)

import pylearn2
from pylearn2 import datasets
from pylearn2.models import autoencoder
from pylearn2.train import Train
from pylearn2.training_algorithms import sgd
from pylearn2.costs import autoencoder
from pylearn2 import termination_criteria

ds=pylearn2.datasets.dense_design_matrix.DenseDesignMatrix(
    X=nts
    )
mdl=pylearn2.models.autoencoder.Autoencoder(
    ds.X_space.dim
    ,ds.X_space.dim
    ,'sigmoid'
    ,'sigmoid'
    )
algo=pylearn2.training_algorithms.sgd.SGD(
    .05
    ,cost=pylearn2.costs.autoencoder.MeanSquaredReconstructionError()#.cost locks up puter!
    ,termination_criterion=termination_criteria.EpochCounter(100)
    ,batch_size=50
    )

trn=pylearn2.train.Train(ds,mdl,algorithm=algo)
#do i need a Y for autoencoder?
