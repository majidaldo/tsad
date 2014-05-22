import numpy as np
import scipy as sp

xs=np.linspace(0,1,1e3)

def sinewave(xs,**kwargs):
    nw=kwargs.setdefault('nw',100) #num waves
    return np.sin(xs*nw*2*np.pi)

def const(xs,**kwargs):
    c=kwargs.setdefault('c',0)
    cs=np.empty(len(xs));cs.fill(c)
    return cs

#generate an even number of waves in the re
nts=np.empty((70,len(xs)),dtype='float32')
for i in xrange(len(nts)):
    nts[i]=sinewave(xs,nw=(i+1)*2)/2+.5 #bc the algo just in rng [0,1]
#nts[-1]=const(xs,c=1)#=1000) #this was messing things up!

import pylearn2
from pylearn2 import datasets
from pylearn2.models import autoencoder
from pylearn2.train import Train
from pylearn2.training_algorithms import sgd
from pylearn2.costs import autoencoder
from pylearn2 import termination_criteria


trn_prop=.7
np.random.shuffle(nts)
ntrn=((len(nts))*float(trn_prop))
nval=len(nts)-ntrn
dst=pylearn2.datasets.dense_design_matrix.DenseDesignMatrix(
    X=nts[:ntrn]
    )
dsv=pylearn2.datasets.dense_design_matrix.DenseDesignMatrix(
    X=nts[:nval]
    )
ds= pylearn2.datasets.dense_design_matrix.DenseDesignMatrix(
    X=nts
)
mdl=pylearn2.models.autoencoder.Autoencoder(
    dst.X_space.dim #input dim
    ,50#ds.X_space.dim #n hidden 
    ,'sigmoid'
    ,'sigmoid'
    ,tied_weights=True
    )
algo=pylearn2.training_algorithms.sgd.SGD(
    .05
    ,cost=pylearn2.costs.autoencoder.MeanSquaredReconstructionError()#.cost locks up puter!
    ,termination_criterion=termination_criteria.MonitorBased(
        channel_name='objective',prop_decrease=.001,N=10)#chnl def in SGD
    #,termination_criterion=termination_criteria.EpochCounter(1000)
    ,monitoring_dataset=ds
    ,batch_size=1#10?100?
    )

from os.path import curdir
from os.path import abspath
trn=pylearn2.train.Train(ds
                         ,mdl
                         ,algorithm=algo
                         ,save_path=abspath(curdir))

from theano import tensor
def train(): return trn.main_loop()
def recon(): return trn.model.reconstruct(tensor.constant(nts[:])).eval()
