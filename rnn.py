import theanets
import numpy as np

import climate
climate.enable_default_logging()

import data
import rnndb


def env(ts_id,**kwargs):
    """use dbts_id='test' kwargs to test things"""
    
    global gts_id
    gts_id=kwargs.setdefault('dbts_id',ts_id)
    
    global trn
    global vld
    global dim_out
    global dim_in
    global noise
    
    ts=data.get(ts_id) 
    tl=int(.75*len(ts)) #potential <-param here
    trn=data.list_call(ts[:tl])
    vld=data.list_call(ts[tl:])
    dim_out=dim_in=data.dim(ts_id)

    noise=np.std(data.get_series(ts_id))*.5 #<- critical param



def make_net(params):
    p=params
    layers=[dim_in]
    for alyr in xrange(params['nl']):
        layers.append( dict(form='lstm' #'rnn'
                            ,size=p['n']
                            ,activation='sigmoid' #ignored on lstm
                            ) )
    layers.append(dim_out)
    #net  = theanets.recurrent.Regressor(
    net = theanets.recurrent.Autoencoder(layers)
    return net



def function(params,run_id=None):
    
    # get network
    
    pc=params.copy()
    pc.pop('iter')
    netfind=list(rnndb.find(gts_id,pc))
    # has a net with these params ever been created?
    if len(netfind)==0:
        net=make_net(pc)
        del netfind
        state                             ='new';                 stateit=0
    else:
        # is there a previous net to resume from?
        lastiters=[int(ait) for ait in rnndb.distinct_iters(gts_id,pc) \
                   if ait<params['iter']]
        if len(lastiters)==0:        state='no previous iter';    stateit=1
        else:
            lastiter=sorted(lastiters)
            lastiter=lastiter[-1]
        
            # chk how many lastiter vs thisiter
            thisiters=list(rnndb.find(gts_id,params))
            pcc=pc.copy()
            pcc['iter']=lastiter
            lastiters=list(rnndb.find(gts_id,pcc))
            nthisiter=len(thisiters)
            nlastiter=len(lastiters)

        
            if nthisiter>=nlastiter: state='no previous iter';    stateit=2
            elif nthisiter<nlastiter:state='previous iter found'; stateit=3
            else: raise Exception('undefined state')

        if state=='previous iter found':
            pcc=pc.copy()
            pcc['iter']=lastiter
            net=rnndb.get_net(gts_id,pcc,i=nthisiter) #'careful! looks good
        elif state=='no previous iter':
            net=make_net(pc)
        else:
            raise Exception('undefined state handler')
    #not elegant but whatever
    print 'stateit',stateit

    xp=theanets.Experiment(net)
    
    xpit=xp.itertrain( trn , vld
                       ,algorithm='rmsprop'
                       ,input_noise=noise
                       #,input_dropout=.3 #idk how this would app here
                       ,nesterov=True
                       #,max_gradient_norm=1
                       ,learning_rate=0.0001 #default
                       #,batch_size=bs
                       #,momentum=0.9
                       ,min_improvement=.005
                       ,patience=5
                       ,validate_every=1
    )

    # assume iter index starts with 0
    if   stateit==0: it=params['iter']+1
    elif stateit==1: it=params['iter']+1
    elif stateit==2: it=params['iter']+1
    elif stateit==3: it=params['iter']-lastiter
    else: raise Exception('undefined state')
    print 'it',it
    import math
    for ait in xrange(it):
        # there is 'err' and 'loss'. mostly the same
        # index 1 is the validation error
        try:
            o= xpit.next()[1]['loss']
            if math.isnan(o):
                raise ValueError('got nan validation')
            rnndb.save_net(gts_id,params,xp.network)
        except StopIteration: pass
            
    rnndb.save_net(gts_id,params,xp.network,run_id=run_id)
    return o #should return the o from the .next() w/o the stopiteration


#todo: have a 'test' ts
def test():
    p={'nl':1,'n':1}
    for ait in [0,0,3,4,5,5]: p['iter']=ait; function(p)
    # function({'nl':1,'iter':0})
    # #function({'n1':1,'iter':0}) # should be a new model
    # function({'nl':1,'iter':3}) # should pick up where left off
    # #function({'n1':1,'iter':4} #shld pick up where left off
    # function({'nl':1,'iter':5}) #shld pick up where left off
    # function({'nl':1,'iter':5}) # shld be new
    
def testseq():
    function({'nl':1,'iter':0})
    function({'nl':1,'iter':1}) # should pick up where left off
    function({'nl':1,'iter':2}) #shld pick up where left off
    function({'nl':1,'iter':1}) # should be new

def testtwo():
    function({'nl':1,'iter':9})
    function({'nl':1,'iter':9}) # new

