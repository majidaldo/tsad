import json
with open('config.json') as cf:
    jl=json.load(cf)
    xpnm=jl['experiment-name']
    db=jl['database']['address']

import sys
import os
sys.path.append(os.path.join('..','..'))
import pomain

def main(job_id,params):
    c=pomain.connect()
    r=pomain.lbv.apply_sync(pomain.main,xpnm
                          ,job_id,params)
    #lbv.get_result(-1) #not working!
    #so getting into the db itself
    import pymongo#3
    mc=pymongo.MongoClient(db)
    ipt=mc['ipython-tasks']['task_records']
    #unicode baby !
    rec=ipt.find_one({u'msg_id':pomain.lbv.history[-1]})
    print rec['stdout']
    print rec['stderr']
    return r
