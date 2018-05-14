from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import Callback
from keras import optimizers
import numpy as np
import tensorflow as tf
import keras.backend as kb

# from Auxilary import TrainErrorPlot, OnlineErrorPlot


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

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

    def train(self, db, state=None):
        trainIn, trainOut = db.getAll()
        #history1 = NBatchLogger()
        history = self.model.fit(trainIn, trainOut, batch_size=64, epochs=50, verbose=0, validation_split=0)
        #self.trainErrHistory.append(history.history['loss'])
        #self.trainErrHistory = self.trainErrHistory+(history.history['loss'])
        #TrainErrorPlot(self.trainErrHistory)
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
        #self.onlineErrHistory.append(err)
        self.plotter.updateOnlineHistory(err) #TODO: Make the list limited in size
        #OnlineErrorPlot(self.onlineErrHistory)

    def deriveAB(self, xk, uk, xk1):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xuk = np.hstack((xk_, uk_))
        target = np.reshape(np.copy(xk1), (1, self.xDim))

        self.model.fit(xuk, target, verbose=0)

        outputLayer = self.model.layers[-1].output
        inputLayer = self.model.layers[0].input

        grad_y0_ = tf.gradients(outputLayer[0][0], inputLayer)
        grad_y1_ = tf.gradients(outputLayer[0][1], inputLayer)
        grad_y2_ = tf.gradients(outputLayer[0][2], inputLayer)
        grad_y3_ = tf.gradients(outputLayer[0][3], inputLayer)
        grad_y4_ = tf.gradients(outputLayer[0][4], inputLayer)
        grad_y5_ = tf.gradients(outputLayer[0][5], inputLayer)
        grad_y6_ = tf.gradients(outputLayer[0][6], inputLayer)
        grad_y7_ = tf.gradients(outputLayer[0][7], inputLayer)


        grad_y = np.zeros((self.xDim, self.xDim + self.uDim))

        tempSess = kb.get_session()
        grad_y[0] = tempSess.run(grad_y0_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[1] = tempSess.run(grad_y1_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[2] = tempSess.run(grad_y2_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[3] = tempSess.run(grad_y3_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[4] = tempSess.run(grad_y4_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[5] = tempSess.run(grad_y5_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[6] = tempSess.run(grad_y6_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        grad_y[7] = tempSess.run(grad_y7_, feed_dict={inputLayer: np.zeros((1, 10))})[0][0]
        #tempSess.close()

        A = np.ones((self.xDim, self.xDim))
        B = np.ones((self.xDim, self.uDim))

        for i in range(self.xDim):
            A[i, :] = grad_y[i][:8]

        for i in range(self.xDim):
            B[i, :] = grad_y[i][8:]

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