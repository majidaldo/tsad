import theanets
import numpy as np

from window import winbatch
import climate
climate.enable_default_logging()

#sgd seeems to always work


#IDEA!! mkae if the optimizer wanst training iter #20 directre,
#just go through the iterations.
#also can make it 'noise' by just making another instance

xp  = theanets.Experiment(
        #theanets.recurrent.Regressor
            theanets.recurrent.Autoencoder(
                 (1
                  #, dict(form='rnn',size=5,activation='relu')
                 , dict(form='lstm',size=5,activation='sigmoid')
                  #, dict(form='lstm',size=10,activation='linear')
                   , 1)
                   )
        #    ,patience=11
        #,train_batches=50
        # ,batch_size=10 #whats the difference? batchsizze=1 doesn't work
            )
new=False
if new == False:
    try: xp=theanets.Experiment('testrnn')
    except: pass


def normalize_seq(seq_batch):
    """shape (T,nsamples,dim)"""
    #make an array of the multidim seq into a more 'standard'
    #shape (nsamples, nfeatures), flattening the mutlidim seq
    #in the process
    #(swapping axes is no prob.. but have to be careful with reshape)
    sb=seq_batch.swapaxes(0,1).reshape(seq_batch.shape[1],-1) 
    #..maybe copying but i don't care
    #now apply a 'normalization'
    from sklearn.preprocessing import normalize as nrm
    sb=nrm(sb,axis=1,copy=False)
    #reshape back
    return sb#sb.reshape(


def add_noise(batch):
    """in shape (sample,T,dim)"""
    batch= batch + np.random.normal(scale=99,size=batch.shape)#,dtype='f32')
    return np.asarray(batch,dtype='f32')
    
#def make_positive
    
#100 samples from callable ..how?

bs=50
#batchproc_callback
ecg=np.loadtxt('ecg.txt',dtype='f32')[::10]
sn=(.2*np.sin(np.linspace(0,3*3.14,num=350-250)))
#put anomaly in input
ecg[250:350]=sn 
tl=int(.7*len(ecg))
ecgb_trn=winbatch( ecg[:tl,None]
                  #,batchproc_callback=add_noise
                    , min_winsize= 50, slide_jump=10, winsize_jump=10
              ,batch_size=bs)
ecgb_val=winbatch( ecg[tl:,None]
                 # ,batchproc_callback=normalize_seq
                    ,min_winsize= 50, slide_jump=10, winsize_jump=10
              ,batch_size=bs)
#e.train(ecgb, ecgb)
#e.network.predict(batch)

xp.train( (ecgb_trn) , (ecgb_val) 
          ,algorithm='rmsprop'
          ,save_progress='testrnn', save_every=1
          ,input_noise=.3
          #,input_dropout=.3
           ,nesterov=True
           ,max_gradient_norm=1
          #,learning_rate=0.0001
          #,batch_size=bs
          #,momentum=0.9
          #,patience=10
          )
#xp.network.save('testrnn')       
         
#it.next()
#xp.train()
import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
sb=ecgb_val()[0]
#chk it didn't just learn some id
#zero a part of it
#zti=int(sb.shape[1]*.25)
#sb[0][None][0][zti:zti*2]=.25
#sb[0][None][0][zti:zti*2]=sn.reshape((zti*2-zti,1))
p=xp.network.predict(sb)
plt.plot(sb[0][None][0]);
plt.plot(p[0][None][0])
