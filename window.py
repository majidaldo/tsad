"""Given a sequence, provides sliding windows of each length as a callable"""


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


def iwin(T, batch_size=32
         , min_winsize= 10, slide_jump=1, winsize_jump=1
         ,max_winsize='T'
         ,winloc_shuffle=True, winsize_shuffle=True):#(seq, batch_size):
    #this is just the logic of the sliding window minibatch
    # yields (windowsize, index of sliding window)
    #just needs length of seq
    #seq=[1,2,3,4,5,6,7,8,9,0]
    #n=len(seq)
    if max_winsize=='T': max_winsize = T #- batch_size + 1 #- slide_jump + 1
    else: T=int(T)

    winsize_rng = ( min_winsize, max_winsize, winsize_jump )
    if winsize_shuffle == True: iws = irandrange(winsize_rng)
    else: iws=(xrange(winsize_rng))
    if winloc_shuffle == True: iwlf = irandrange
    else: iwlf=xrange
    for winsize in iws:
        iwl = iwlf((0,T,slide_jump))
        winlocs=[]
        for winloc in iwl:
            if (T-winsize)<winloc: continue
            winlocs.append(winloc)
            if len(winlocs)==batch_size:
                yield winsize , winlocs
                winlocs=[]
         #the remainding from the location looping
        if len(winlocs)!=0: yield winsize,winlocs

    
