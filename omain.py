import rnn
import numpy as np
import math


#rnn.env('sleep')

def main(job_id, params):

    

    params['iter']=[itermap(params['iter'])]
    no_array={}
    # for some reason params are arrays!!
    for aparam in params:
        no_array[aparam]=params[aparam][0]
    
    
    o=float(rnn.function(no_array,run_id=job_id))
    if math.isnan(o): raise ValueError('got nan result')

    
    return {'main':o}


def rootmap(y,scalex,p=.5):
    """map 0-1(y:x) to some (square?) root function (x->y)"""
    return ((scalex**p)*y)**(p**-1)


def itermap(y,mostiter=20):
    iterlim=99999
    
    if y==1: y=.99999999
    o=int(math.ceil(rootmap(y,mostiter)))
    if o==mostiter: return iterlim
    else: return o
