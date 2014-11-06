"""has functions to handle segmenting a long timeseries for consumption
in a neural network"""


from random import randrange


def irandrange(*rng):
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


def iwin(T, batch_size=32
         , min_winsize= 10, slide_jump=1, winsize_jump=1
         ,max_winsize='T'
         ,winloc_shuffle=True, winsize_shuffle=True):
    """this is just the (separated out) logic of the 
    sliding window minibatch
    returns: (windowsize, indexes of sliding window) """
    if max_winsize=='T': max_winsize = T
    else: T=int(T)

    winsize_rng = ( min_winsize, max_winsize, winsize_jump )
    if winsize_shuffle == True: iws = irandrange(*winsize_rng)
    else: iws=xrange(*winsize_rng)
    if winloc_shuffle == True: iwlf = irandrange
    else: iwlf=xrange
    for winsize in iws:
        iwl = iwlf(*(0,T,slide_jump))
        winlocs=[]
        for winloc in iwl:
            if (T-winsize)<winloc: continue
            winlocs.append(winloc)
            if len(winlocs)==batch_size:
                yield winsize , winlocs
                winlocs=[]
         #the remainig from the location looping
        if len(winlocs)!=0: yield winsize,winlocs

def winbatch(seq, batch_igen=iwin, **kwargs):
    """first axis of sequence should be time"""
    bi=batch_igen(len(seq),**kwargs)
    for awinsize , iwinlocs in bi:
        abatch=[]
        for awinloc in iwinlocs:
            abatch.append(seq[awinloc:awinloc+awinsize])
        yield abatch