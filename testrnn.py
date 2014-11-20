import theanets
import numpy as np

from window import winbatch
import climate
climate.enable_default_logging()



e = theanets.Experiment(
    #theanets.recurrent.Regressor
    theanets.recurrent.Autoencoder
    ,layers=(1, 10, 1)
    ,train_batches=16
    #,batch_size=16 #whats the difference?
    )

ecgb=winbatch(np.loadtxt('ecg.txt',dtype='f32')[:,None]
              ,batch_size=e.args.batch_size)
#e.run(ecgb, ecgb)
#e.network.predict(batch)

