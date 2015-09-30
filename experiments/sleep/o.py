import sys
sys.path.append('../..')
from omain import *
import rnn
import numpy as np
import math


import json
with open('config.json') as cf:
    xpnm=json.load(cf)['experiment-name']


rnn.env(xpnm) 


