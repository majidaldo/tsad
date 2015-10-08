import pymongo
con=pymongo.MongoClient('localhost',27017)
db=con.drop_database('rnn')
db=con['rnn']

from tempfile import NamedTemporaryFile as ntf
import os

import pickle
from bson.binary import Binary
import theanets

_mydir,_=os.path.split(os.path.realpath(__file__))

def get_net(ts_id
            ,params
            ,i=-1): # n -1 gets the last one interted
    """get str rep from db and put it into a file"""
    tbl=col=db[ts_id] #'collection'
    
    pc=params.copy()
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
    pc=params.copy()

    tfp=ntf(dir=_mydir,suffix='.tmp',delete=False)
    try:
        net.save(tfp.name);tfp.close()
        pc['net']=pickle.dumps(net.load(tfp.name))
        tfp.close()
    finally: #cleanup
        os.remove(tfp.name)

    if run_id != None: pc['run_id']=run_id
    
    o= tbl.insert_one(pc).inserted_id
    return o



