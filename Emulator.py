from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.callbacks import Callback
from keras import optimizers
import numpy as np
class Emulator():
    def __init__(self, new=False,filePath=''):
        if new:
            self.model = Sequential()
            self.model.add(Dense(300, input_dim=10, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(400, input_dim=300, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(300, input_dim=400, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(200, input_dim=300, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(8, kernel_initializer='normal'))
            self.optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            from keras.utils import plot_model
            plot_model(self.model, to_file='modelNew6xy.png', show_shapes=True, show_layer_names=True)
            self.compile()
        else:
            self.restoreModel(filePath)


    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
    def train(self, db,state=None):
        input, target = db.getAll()
        history = NBatchLogger()

        self.model.fit(input, target, batch_size=100, epochs=100, verbose=2,
                 validation_split=0.3)
        return


    def predict(self, xk,uk):
        xk_ = np.reshape(np.copy(xk), (1,8))
        uk_ = np.reshape(np.copy(uk), (1,2))
        xk_uk = np.hstack((xk_, uk_))
        return self.model.predict(xk_uk)

    def saveModel(self, filePath):
        self.model.save(filePath)

    def restoreModel(self, filePath):
        self.model =load_model(filePath)


class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display=100):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()