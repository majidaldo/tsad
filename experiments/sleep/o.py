import sys
sys.path.append('../..')
from omain import *


import json
with open('config.json') as cf:
    xpnm=json.load(cf)['experiment-name']


env(xpnm) 


