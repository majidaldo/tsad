import theanets
import numpy as np

from window import winbatch
import climate
climate.enable_default_logging()


import inspect
import os
fp=os.path.split(os.path.realpath(__file__))[0];
os.chdir(fp)
import dataset as ds

dbfp='../files/rnn.sqlite'
dbp='sqlite:///'+dbfp
test=True
if test==True:
    try: os.remove(dbfp)
    except: pass

db=ds.connect(dbp)
tbl=db['rnns']

#IDEA!! mkae if the optimizer wanst training iter #20 directre,
#just go through the iterations.
#also can make it 'noise' by just making another instance

# n each layer
# n iter
# n 


def get_net(params,i=-1): # n -1 gets the last one interted
    """get str rep from db and put it into a file"""
    pc=params.copy()
    found=list(tbl.find(**pc))
    #if len(found)>1: raise Exception('more than one model matched params')
    if len(found)==0: return None
    found=found[i] # get's the last one inserted

    try:
        tfp=ntf(dir=fp,suffix='.tmp',delete=False)
        with open(tfp.name,'wb') as f: f.write(found['net'])
        tfp.close()
        net=theanets.Network.load(tfp.name)
    finally: #cleanup
        os.remove(tfp.name)
    return net

from tempfile import NamedTemporaryFile as ntf
def save_net(params,net):
    
    pc=params.copy()
    #nn=len(list(tbl.find(**pc)))
    #if nn>1: raise Exception('more than one model matched params')

    try:
        tfp=ntf(dir=fp,suffix='.tmp',delete=False)
        net.save(tfp.name)
        with open(tfp.name,'rb') as f: pc['net']=buffer(f.read())
        tfp.close()
    finally: #cleanup
        os.remove(tfp.name)

    #if   nn==0: to=tbl.insert(pc)
    #elif nn==1: to=tbl.update(pc,pc.keys()) #returns false
    return tbl.insert(pc)


bs=50
#batchproc_callback
ecg=np.loadtxt('ecg.txt',dtype='f32')[::10]
sn=(.2*np.sin(np.linspace(0,3*3.14,num=350-250)))
#put anomaly in input
ecg[250:350]=sn
tl=int(.7*len(ecg))
ecgb_trn=winbatch( ecg[:tl,None]
                    , min_winsize= 50, slide_jump=10, winsize_jump=10
              ,batch_size=bs)
ecgb_val=winbatch( ecg[tl:,None]
                    ,min_winsize= 50, slide_jump=10, winsize_jump=10
              ,batch_size=bs)


# todo possible to have n1=0 nx=0 by just not having it there
def make_net(params):
    p=params
    #net  = theanets.recurrent.Regressor(
    net = theanets.recurrent.Autoencoder(
            (1
             #, dict(form='rnn',size=5,activation='relu')
             , dict(form='lstm',size=p['n1'],activation='sigmoid')
             #, dict(form='lstm',size=10,activation='linear')
             , 1)
    )
    return net


# todo make trn and val rnd


def function(params):

    # get network
    
    pc=params.copy()
    pc.pop('iter')
    netfind=list(tbl.find(**pc))
    # has a net with these params ever been created?
    if len(netfind)==0:
        net=make_net(pc)
        del netfind
        state='new';                                            stateit=0
    else:
        # is there a previous net to resume from?
        lastiters=[arow['iter'] for arow in tbl.distinct('iter',**pc)]
        lastiter=sorted(lastiters)
        lastiter=lastiter[-1]
        # chk how many lastiter vs thisiter
        thisiters=list(tbl.find(**params))
        pcc=pc.copy()
        pcc['iter']=lastiter
        lastiters=list(tbl.find(**pcc))
        nthisiter=len(thisiters)
        nlastiter=len(lastiters)

        if len(lastiters)==0:      state='no previous iter';    stateit=1
        elif nthisiter>=nlastiter: state='no previous iter';    stateit=2
        elif nthisiter<nlastiter:  state='previous iter found'; stateit=3
        else: raise Exception('undefined state')

        if state=='previous iter found':
            pcc=pc.copy()
            pcc['iter']=lastiter
            net=get_net(pcc,i=nthisiter) #'careful! looks good
        elif state=='no previous iter':
            net=make_net(pc)
        else:
            raise Exception('undefined state handler')
    #not elegant but whatever
    print 'stateit',stateit

    xp=theanets.Experiment(net)
    
    xpit=xp.itertrain( (ecgb_trn) , (ecgb_val) 
                       ,algorithm='rmsprop'
                       #,save_progress='testrnn', save_every=1
                       ,input_noise=.3
                       #,input_dropout=.3
                       ,nesterov=True
                       ,max_gradient_norm=1
                       ,validate_every=1
                       #,learning_rate=0.0001
                       #,batch_size=bs
                       #,momentum=0.9
                       #,patience=10
    )


    # assume iter index starts with 0
    if   stateit==0: it=params['iter']+1
    elif stateit==1: it=params['iter']+1
    elif stateit==2: it=params['iter']+1
    elif stateit==3: it=params['iter']-lastiter
    else: raise Exception('undefined state')
    for ait in xrange(it):
        # there is 'err' and 'loss'. mostly the same
        # index 1 is the validation error
        o= xpit.next()[1]['loss']
    save_net(params,xp.network)
    return o

# maybe the obj function should be for validation set
# todo see what to do with validation


def test():
    function({'n1':1,'iter':0})
    #function({'n1':1,'iter':0}) # should be a new model
    function({'n1':1,'iter':3}) # should pick up where left off
    #function({'n1':1,'iter':4} #shld pick up where left off
    function({'n1':1,'iter':5}) #shld pick up where left off
    function({'n1':1,'iter':5}) # shld be new
    
def testseq():
    function({'n1':1,'iter':0})
    function({'n1':1,'iter':1}) # should pick up where left off
    function({'n1':1,'iter':2}) #shld pick up where left off
    function({'n1':1,'iter':1}) # should be new

def testtwo():
    function({'n1':1,'iter':9})
    function({'n1':1,'iter':9}) # new

