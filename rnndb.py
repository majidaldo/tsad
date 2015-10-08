import pymongo
con=pymongo.MongoClient('localhost',27017)
#db=con.drop_database('rnn')
db=con['rnn']

from tempfile import NamedTemporaryFile as ntf
import os

import pickle
from bson.binary import Binary
import theanets #todo: set a version/commit

_mydir,_=os.path.split(os.path.realpath(__file__))


def find(ts_id
        ,params):
    tbl=db[ts_id]
    pc=tomongotypes(params)
    return tbl.find(pc)

def distinct_iters(ts_id
                   ,params):
    found=find(ts_id,params)
    return found.distinct('iter')

def get_net(ts_id
            ,params
            ,i=-1): # n -1 gets the last one interted
    """get str rep from db and put it into a file"""
    tbl=col=db[ts_id] #'collection'
    
    pc=tomongotypes(params)
    found=list(tbl.find(pc))
    #if len(found)>1: raise Exception('more than one model matched params')
    if len(found)==0: return None
    found=found[i] # get's the last one inserted

    tfp=ntf(dir=_mydir,suffix='.tmp',delete=False)
    try:
        with open(tfp.name,'wb') as f: f.write(found['net'])
        tfp.close()
        net=theanets.Network.load(tfp.name)
    finally: #cleanup
        os.remove(tfp.name)
    return net



def save_net(ts_id
             ,params,net
             ,run_id=None):
    tbl=col=db[ts_id] #'collection'

    if run_id != None: pc['run_id']=run_id
    pc=tomongotypes(params)
    
    tfp=ntf(dir=_mydir,suffix='.tmp',delete=False)
    try:
        net.save(tfp.name);tfp.close()
        pc['net']=pickle.dumps(net.load(tfp.name))
        tfp.close()
    finally: #cleanup
        os.remove(tfp.name)
        
    o= tbl.insert_one(pc).inserted_id
    return o



def tomongotypes(params):
    mt={}
    for ap in params:
        try: #chk for number
            params[ap]/1.0
            #just using ints but could use floats for flexibility
            mt[ap]=float(params[ap]) 
        except:
            mt[ap]=params[ap]

    return mt
    
# todo: climate loggind stdout
