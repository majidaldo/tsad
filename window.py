"""Given a sequence, provides sliding windows of each length as a callable"""

from itertools import islice

def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

from random import randrange
# implement its own shuffling
# shuffle batches (winsize) and shuffle minibatches (batchsize)

#need to gen window sizes range(,,)
#need to gen intervals over all the ts

def irandrange(rng):
    #should be a primitive
    """gives a random number in rng until numbers are exhausted"""
    try: rng=int(rng)
    except: pass
    try: len(rng)
    except TypeError: rng=(rng,)
    picked=set()
    while len(picked) < len(xrange(*rng)):
        pick=randrange(*rng)
        if pick not in picked: 
            picked.add(pick)
            yield pick


#each window size and window jump determines how many windows can be made
#find common number of windows for 

def iwin(win_shuffle=True, winsize_shuffle=True):#(seq, batch_size):
    # yields (windowsize, index)
    #just needs length of seq
    seq=[1,2,3,4,5,6,7,8,9,0]
    n=len(seq)
    batch_size = 3
    min_winsize = 2
    slide_jump = 1
    #max_winsize = n - batch_size + 1 #- slide_jump + 1

    winsize_rng=(min_winsize,max_winsize,1)
    if winsize_shuffle==True: iwsr=irandrange(*win_rng)
    else: iwsr=(xrange(*win_rng))
    win_rng

    
