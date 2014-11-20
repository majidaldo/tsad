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

import numpy.random as rng
def batches(args, X, horizon=None):
    if horizon is None:
        horizon = args.pool_error_start + 2
    batch = np.zeros((horizon, args.batch_size, args.layers[0]), 'f')
    def create_batch():
        for b in range(args.batch_size):
            x = X[rng.randint(len(X))]
            i = rng.randint(len(x) - horizon)
            batch[:, b] = x[i:i + horizon]
        return [batch]
    return create_batch