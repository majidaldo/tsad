import json
with open('config.json') as cf:
    xpnm=json.load(cf)['experiment-name']

import sys
import os
sys.path.append(os.path.join('..','..'))
import pomain

def main(job_id,params):
    c=pomain.connect()
    r=pomain.lbv.apply_sync(pomain.main,xpnm
                          ,job_id,params)
    #load balanced view: just one engine gets it
    #r=pomain.lbv.apply_sync(lambda x:  float(x),1)
    return r
