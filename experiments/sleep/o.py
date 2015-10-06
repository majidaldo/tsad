import json
with open('config.json') as cf:
    xpnm=json.load(cf)['experiment-name']

import sys
sys.path.append('../..')
from omain import *

import os
os.chdir(os.path.join('..','..'))
env(xpnm)
