import theanets
import numpy as np
from theanets.dataset import SequenceDataset as DS
from window import winbatch
import climate
climate.enable_default_logging()



xp = theanets.Experiment(
    #theanets.recurrent.Regressor
    theanets.recurrent.Autoencoder
    ,layers=(1, 10, 1)
    ,patience=11
    #,train_batches=50
    ,batch_size=10 #whats the difference? batchsizze=1 doesn't work
    )

def normalize_seq(seq_batch):
    """shape (T,nsamples,dim)"""
    #make an array of the multidim seq into a more 'standard'
    #shape (nsamples, nfeatures), flattening the mutlidim seq
    #in the process
    #(swapping axes is no prob.. but have to be careful with reshape)
    sb=seq_batch.swapaxes(0,1).reshape(seq_batch.shape[1],-1) 
    #..maybe copying but i don't care
    #now apply a 'normalization'
    from sklearn.preprocessing import normalize as n
    sb=n(sb,axis=1,copy=False)
    #reshape back
    return sb#sb.reshape(

#batchproc_callback
ecg=np.loadtxt('ecg.txt',dtype='f32')
tl=int(.7*len(ecg))
ecgb_trn=winbatch( ecg[:tl,None]
                  ,batchproc_callback=normalize_seq
                    , min_winsize= 500, slide_jump=10, winsize_jump=100
              ,batch_size=xp.args.batch_size)
ecgb_val=winbatch( ecg[tl:,None]
                  ,batchproc_callback=normalize_seq
                    ,min_winsize= 500, slide_jump=10, winsize_jump=100
              ,batch_size=xp.args.batch_size)
#e.train(ecgb, ecgb)
#e.network.predict(batch)

