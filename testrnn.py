import theanets
import numpy as np

from window import winbatch


e = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(1, 10, 1)
    ,train_batches=16
    #,batch_size=16 #whats the difference?
    )

ecgb=winbatch(np.loadtxt('ecg.txt',dtype='f32')[:,None]
              ,batch_size=e.args.batch_size)
#e.run(ecgb, ecgb)
#predict(batch)