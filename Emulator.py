from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import Callback
import keras
from keras import optimizers
import numpy as np
import tensorflow as tf
import keras.backend as kb

# from Auxilary import TrainErrorPlot, OnlineErrorPlot
import getpass
import os
import glob

username = getpass.getuser()
modelDir = '/home/' + username + '/PycharmProjects/Reacher/Networks/'

class Emulator:

    def __init__(self, plotter, new=False, filePath=''):
        self.xDim = 8
        self.uDim = 2
        self.plotter = plotter


        if new:
            self.model = Sequential()
            self.model.add(Dense(100, input_dim=10, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(150, input_dim=100, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(self.xDim, kernel_initializer='normal'))
            self.optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            self.compile()

        else:
            self.restoreModel(filePath)

        # For obtaining gradients
        self.outputLayer = self.model.layers[-1].output
        self.inputLayer = self.model.layers[0].input
        self.grad_y0_ = tf.keras.backend.gradients(self.outputLayer[0][0], self.inputLayer)[0]
        self.grad_y1_ = tf.keras.backend.gradients(self.outputLayer[0][1], self.inputLayer)[0]
        self.grad_y2_ = tf.keras.backend.gradients(self.outputLayer[0][2], self.inputLayer)[0]
        self.grad_y3_ = tf.keras.backend.gradients(self.outputLayer[0][3], self.inputLayer)[0]
        self.grad_y4_ = tf.keras.backend.gradients(self.outputLayer[0][4], self.inputLayer)[0]
        self.grad_y5_ = tf.keras.backend.gradients(self.outputLayer[0][5], self.inputLayer)[0]
        self.grad_y6_ = tf.keras.backend.gradients(self.outputLayer[0][6], self.inputLayer)[0]
        self.grad_y7_ = tf.keras.backend.gradients(self.outputLayer[0][7], self.inputLayer)[0]
        #tf.get_default_graph().finalize()

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

    def train(self, db, state=None):
        trainIn, trainOut = db.getAll()
        history = self.model.fit(trainIn, trainOut, batch_size=64, epochs=50, verbose=0, validation_split=0)
        self.plotter.updateTrainingHistory(history.history['loss']) #TODO: Make the list limited in size

    def predict(self, xk, uk):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xk_uk = np.hstack((xk_, uk_))
        return self.model.predict(xk_uk)

    def saveModel(self, filePath):
        self.model.save(filePath)

    def restoreModel(self, filePath):
        self.model = load_model(filePath)

    def evaluatePredictionError(self,xk,uk,xk1):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xuk = np.hstack((xk_, uk_))
        target = np.reshape(np.copy(xk1), (1, self.xDim))
        err = self.model.evaluate(x=xuk, y=target, batch_size=100,verbose=0)
        self.plotter.updateOnlineHistory(err) #TODO: Make the list limited in size

    def deriveAB(self, xk, uk, xk1):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xuk = np.hstack((xk_, uk_))
        target = np.reshape(np.copy(xk1), (1, self.xDim))

        #self.model.fit(xuk, target, batch_size=1, epochs=1, verbose=0, validation_split=0)
        #self.model.evaluate(x=xuk, y=target, batch_size=100,verbose=0)

        sess = kb.get_session()
        grad_y = sess.run([self.grad_y0_, self.grad_y1_, self.grad_y2_, self.grad_y3_, self.grad_y4_, self.grad_y5_, self.grad_y6_, self.grad_y7_], feed_dict={self.inputLayer: xuk})

        A = np.zeros((self.xDim, self.xDim))
        B = np.zeros((self.xDim, self.uDim))

        for i in range(self.xDim):
            A[i, :] = grad_y[i][0][:8]

        for i in range(self.xDim):
            B[i, :] = grad_y[i][0][8:]

        return A, B






'''class NBatchLogger(Callback):
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
            self.metric_cache.clear()'''