import rnn
import numpy as np

def main(job_id, params):

    no_array={}
    # for some reason params are arrays!!
    for aparam in params:
        no_array[aparam]=params[aparam][0]

    #return 1.0
    return {'main':float(rnn.function(no_array))}
