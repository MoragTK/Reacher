from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
import numpy as np
import tensorflow as tf
import keras.backend as kb
import getpass

username = getpass.getuser()
modelDir = '/home/' + username + '/PycharmProjects/Reacher/Networks/'

'''
    This Class implements the learned model in the algorithm.
    It implements the model using a neural network, and provides functions 
    that allow to initialize it, train it, use it for prediction, 
    calculate the prediction error, derive it according to the inputs.
'''


class Emulator:

    # Constructor
    def __init__(self, dataBase, plotter, new=False, filePath=''):
        self.xDim = 8
        self.uDim = 2
        self.db = dataBase
        self.plotter = plotter

        self.minTrainError = 100

        if new:
            self.model = Sequential()
            self.model.add(Dense(100, input_dim=10, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(150, input_dim=100, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(self.xDim, kernel_initializer='normal'))
            self.optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

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

    # Trains the net on the samples from the data base db
    # Updates the training error in the training error graph.
    def train(self):
        trainIn, trainOut = self.db.getAll()
        history = self.model.fit(trainIn, trainOut, batch_size=64, epochs=50, verbose=0, validation_split=0)
        self.plotter.updateTrainingHistory(history.history['loss'])
        self.minTrainError = min(history.history['loss'])

    # Receives the networks input - current state x[k] and action u[k], and predicts the next state x[k+1]
    def predict(self, xk, uk):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xk_uk = np.hstack((xk_, uk_))
        return self.model.predict(xk_uk)

    # Save the model in the given path.
    def saveModel(self, filePath):
        self.model.save(filePath)

    # Restore a model from the given path.
    def restoreModel(self, filePath):
        self.model = load_model(filePath)

    # Evaluates the prediction error for x[k], u[k], given the true value of x[k+1]
    # Updates the prediction error in the online error graph.
    def evaluatePredictionError(self, xk, uk, xk1):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xuk = np.hstack((xk_, uk_))
        target = np.reshape(np.copy(xk1), (1, self.xDim))
        err = self.model.evaluate(x=xuk, y=target, batch_size=64, verbose=0)
        self.plotter.updateOnlineHistory(err)

    # Derives the model (Partial derivative) according to the inputs x[k] and u[k].
    def deriveModel(self, xk, uk):
        xk_ = np.reshape(np.copy(xk), (1, self.xDim))
        uk_ = np.reshape(np.copy(uk), (1, self.uDim))
        xuk = np.hstack((xk_, uk_))

        sess = kb.get_session()
        grad_y = sess.run([self.grad_y0_, self.grad_y1_, self.grad_y2_, self.grad_y3_, self.grad_y4_, self.grad_y5_,
                           self.grad_y6_, self.grad_y7_],
                          feed_dict={self.inputLayer: xuk})

        A = np.zeros((self.xDim, self.xDim))
        B = np.zeros((self.xDim, self.uDim))
        for i in range(self.xDim):
            A[i, :] = grad_y[i][-1][:8]

        for i in range(self.xDim):
            B[i, :] = grad_y[i][-1][8:]

        return A, B
