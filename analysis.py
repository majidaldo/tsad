import pandas as pd
import numpy as np
import os

import config as configmod
from config import config
# get data from db
    
from spearmint.utils.database.mongodb import MongoDB
mdb=MongoDB(config['rnndb']) #samd db as rnns



def get_runs(xpnm):
    jobs=mdb.load(xpnm,'jobs',{'status':'complete'})
    try: jobs[0]
    except KeyError: jobs=[jobs]
    finally: params=jobs[0]['params'].keys()

    data=[]
    for ajb in jobs:
        arow=[]
        for ap in params:
            dd=ajb['params'][ap]['values'][0]
            dt=ajb['params'][ap]['type'][0]
            arow.append(np.array(dd,dtype=dt))
        arow.append(ajb['values']['main'])
        arow.append((ajb['id']))
        data.append(tuple(arow))

    columns=params[:]
    columns.append('o')
    columns.append('run_id')
    runs=pd.DataFrame(data=data,columns=columns)
    return runs


def get_best_params(xpnm):
    #todo what if muliple nets with same 'params'?
    runs=get_runs(xpnm)
    bp=dict(runs.ix[runs['o'].idxmin()])
    bp.pop('o')
    best_params={}
    for ap in (bp):
        try: #chk for number
            bp[ap]/1.0
            best_params[ap]=float(bp[ap])
        except:
            best_params[ap]=bp[ap]
    return best_params

import rnndb
import omain
def get_best_net(xpnm):
    #assert(len(list(tbl.find(**best_params)))==1)
    params=get_best_params(xpnm)
    params['iter']=omain.itermap(params['iter'])
    return rnndb.get_net(xpnm,params)


# import matplotlib.pyplot as plt
# import data
# #ts=data.get(ts_id) #,length=100) len should ~250
# #tl=int(.7*len(ts))
# #trn=(ts[:tl])
# #vld=(ts[tl:])

# #def diag(ts=trn,i=0):
# #    plt.plot(ts[i])
# #plt.plot(get_best_net().predict(ts)[i])

#ts=slidingwin size= step=
import sklearn.metrics as metrics
def get_errs(wints,net):
    p=net.predict(wints)
    errs=[]
    for i in xrange(wints.shape[0]):
        errs.append(metrics.mean_squared_error(wints[i,:,0],p[i])**1 )
    return errs


from pandas import rolling_apply
import data
def errs(ts_id,win,**kwargs):
    ts=data.get_series(ts_id)[:,0]
    tsdf=pd.Series(ts)
    bn=get_best_net(ts_id)
    mse=lambda win:np.mean(
        bn.predict(np.array(win,dtype='float32')[:,None,None])
        -win
    )**2
    if win==0: #no window. just return all errors at once
        pr= (bn.predict(ts[:,None,None])[:,0,0]-ts)**2;
        return pr
    return \
        rolling_apply(tsdf
                      ,win
                      ,mse
                      ,center=True
        )
    

#bodiag rng nl and n
def bo_diag(ts_id):
    d=get_runs(ts_id)
    for ar in d['run_id']:
        if 'patience elapsed' in get_log(ts_id,ar):
            ri=d[d['run_id']==ar].index;
            if  d.loc[ri,'iter'].any()==1: pass
            else:
                if 'patience elapsed' in get_log(ts_id,ar):
                    d.loc[ri,'iter']=1
                else: pass
    d=d[d['iter']==1] #just get the ones that i'm sure patience elapsed
    d=d[d.columns.drop(['run_id','iter'])] #no need
    # still has objects instead of elems of a dtype
    d['n']=np.array(d['n'],dtype=np.int)
    d['nl']=np.array(d['nl'],dtype=np.int)
    d=d.sort_values(by=['nl','n'])
    return d


def get_log(ts_id,run_id):
    thisdir=os.path.split(os.path.abspath(configmod.__file__))[0]
    run_id=str(run_id)
    fn= '0'*(8-len(run_id))+run_id+'.out'
    fn= (os.path.join(thisdir,'experiments',ts_id,'output',fn))
    return open(fn).read()




#todo chk for patience in log 'patience elapsed'

