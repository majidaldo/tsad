import numpy as np
from window import winbatch

import syn

import os

def get_series(id):
    """returns 2d shape (time,ndim)"""
    
    def txtrdr(*args,**kwargs):
        kwargs.setdefault('dtype','f32')
        return np.loadtxt(os.path.join('data',id),**kwargs)
    
    if 'ecg' in id:
        ecg=txtrdr()[::20,None] 
        if id=='ecg':
            return ecg
        elif id=='ecg-anom':
            ecg=get_series('ecg')
            sn1=.25*ecg.shape[0]
            sn2= .3*ecg.shape[0]
            sn=(.2*np.sin(np.linspace(0,3*3.14,num=(sn2-sn1))))
            #put anomaly in input
            ecg[sn1:sn2]=sn[:,None]
            return ecg
        
    elif 'sleep'==id:
        #from https://physionet.org/atm/ucddb/ucddb002_lifecard.edf/180/60/rdsamp/csv/pS/samples.csv
        return txtrdr(id,skiprows=2,delimiter=',')[::3,2,None]

    elif 'sin'==id:
        #import syn why not work here???
        return syn.pulsegen()[:,None]

    elif 'spike'==id:
        return syn.cyclespike()

    #should not be here    
    raise KeyError('series not found')



        
def get_kwargs(id,**kwargs):
    # nice to check the number of batches
    # winbatch(ts,getKwargs(ts)).length should be 'reasonable'
    
    kwargs2=kwargs.copy()
    
    ts=get_series(id)
    tnth=int(.1*len(ts))
    kwargs.setdefault('min_winsize',        int(    tnth))
    kwargs.setdefault('slide_jump' ,        int(.10*tnth))
    kwargs.setdefault('winsize_jump',       int(.10*tnth))
    kwargs.setdefault('batch_size',         10           )
    
    if 'ecg'==id:
        kwargs['min_winsize']=  300
        kwargs['slide_jump']=   20
        kwargs['winsize_jump']= 20
        kwargs['batch_size']=   30

    elif 'sleep'==id:
        kwargs['min_winsize']=  300
        kwargs['slide_jump']=   20
        kwargs['winsize_jump']= 20
        kwargs['batch_size']=   30

    elif 'sin'==id:
        kwargs['batch_size']=   30

    #else:
    #    raise KeyError

    # but the kwargs in the func arg overrides
    for ak in kwargs2: kwargs[ak]=kwargs2[ak]
            
    return kwargs



def get(id,**kwargs):
    """for consumption by rnn training"""
    
    ts=get_series(id)
    kwargs=get_kwargs(id,**kwargs)
    
    batches=[]
    bg=winbatch(ts,**kwargs)
    for i in xrange(bg.length):
        batches.append(bg())
        
    return batches



def dim(id):
    return get_series(id).shape[1]



from window import slidingwindow as win
def window(id,**kwargs):
    a=[]
    ts=get_series(id)
    for awin in win(ts,**kwargs):
        a.append(awin)
    return np.array(a,dtype='f32')



from itertools import cycle
class list_call(object):

    def __init__(self,tsbatch_list,**kwargs):
        self.kwargs=kwargs
        self.tsbatch_list=tsbatch_list
        self.iter=cycle(self.__iter__())

    def __iter__(self):
        return iter(self.tsbatch_list)

    def __call__(self):
        return self.iter.next()

    def __len__(self):
        return len(self.tsbatch_list)
        
