import numpy as np
from window import winbatch

#todo make trning and vld data rnd

def get_series(id,**kwargs):
    """returns 2d shape (time,ndim)"""
    if 'ecg' in id:
        ecg=np.loadtxt('ecg.txt',dtype='f32')[::10][:,None]
        tl=int(.7*len(ecg))
        if id=='ecg':
            return ecg
        elif id=='ecg-trn':
            ecg=get_series('ecg')
            sn=(.2*np.sin(np.linspace(0,3*3.14,num=350-250)))
            #put anomaly in input
            ecg[550:650]=sn[:,None]
            return ecg[:tl]
        elif id=='ecg-vld':
            return ecg[tl:]
        else:
            raise ValueError('series not found')

        
def get(id,**kwargs):
    """for consumption by rnn training"""
    ts=get_series(id,**kwargs)
    if 'ecg' in id:
        kwargs.setdefault('min_winsize',200)
        kwargs.setdefault('slide_jump',10)
        kwargs.setdefault('winsize_jump',10)
        kwargs.setdefault('batch_size',50)
        return winbatch(ts,**kwargs)


def dim(id):
    if 'ecg' in id:
        return get('ecg-trn').mybatch_gen.next().shape[2]


from window import window as win
def window(id,**kwargs):
    a=[]
    ts=get_series(id,**kwargs)
    for awin in win(ts,**kwargs):
        a.append(awin)
    return np.array(a,dtype='f32')
