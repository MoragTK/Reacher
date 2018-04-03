import tensorflow as tf
import numpy as np
import getpass

# Username for storing and loading purposes
username = getpass.getuser()
modelDir = '/home/' + username + '/PycharmProjects/Reacher/net/'
modelPath = '/home/' + username + '/PycharmProjects/Reacher/net/emulator'


class Emulator:

    def __init__(self):

        # Dimensions
        self.inputDim = 10
        self.Layer1Dim = 50
        self.Layer2Dim = 14
        self.Layer3Dim = 25
        self.outputDim = 8

        # Learning algorithm parameters
        self.batch = 100  # batch size
        self.epochs = 20  # number of epochs
        self.rate = 0.01  # learning rate

        # place holders for training data sets
        self.xuIn = tf.placeholder('float', [None, self.inputDim])
        self.xOutReal = tf.placeholder('float', [None, self.outputDim])

        # weights dictionary
        self.W = {
                'w0': tf.Variable(tf.truncated_normal([self.inputDim, self.Layer1Dim])),
                'w1': tf.Variable(tf.truncated_normal([self.Layer1Dim, self.Layer2Dim])),
                'w2': tf.Variable(tf.truncated_normal([self.Layer2Dim, self.Layer3Dim])),
                'w3': tf.Variable(tf.truncated_normal([self.Layer3Dim, self.outputDim]))
        }

        # biases dictionary
        self.b = {
                'b0': tf.Variable(tf.truncated_normal([self.Layer1Dim])),
                'b1': tf.Variable(tf.truncated_normal([self.Layer2Dim])),
                'b2': tf.Variable(tf.truncated_normal([self.Layer3Dim])),
                'b3': tf.Variable(tf.truncated_normal([self.outputDim]))
        }

        # initializing all variables and creating a saver for the model
        self.saver = tf.train.Saver()
        self.xOut = self.emulatorNetwork(self.xuIn, self.W, self.b)
        self.costFunc = tf.losses.mean_squared_error(self.xOutReal, self.xOut)
        self.trainOp = tf.train.AdamOptimizer(self.rate).minimize(self.costFunc)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # a flag that tells us whether the model is a restored one. if restored once, no need to restore it again.
        self.restored = False

    # Defining the TF neural network graph
    def emulatorNetwork(self, dataIn, W, b):

        layer_1 = tf.add(tf.matmul(dataIn, W['w0']), b['b0'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, W['w1']), b['b1'])
        layer_2 = tf.nn.relu(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, W['w2']), b['b2'])
        layer_3 = tf.nn.relu(layer_3)

        layer_4 = tf.add(tf.matmul(layer_3, W['w3']), b['b3'])

        return layer_4

    # Train model with data that is currently in the data base.
    def train(self, db, state):
        if state == 'TRAIN' and (self.restored is False):
            self.saver = tf.train.import_meta_graph(modelPath + '.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(modelDir))
            print "Model was restored from the following path: " + modelPath
            self.restored = True

        trainIn, trainOut = db.getAll()
        for epoch in range(self.epochs):
            _, cost = self.sess.run([self.trainOp, self.costFunc], feed_dict={self.xuIn: trainIn, self.xOutReal: trainOut})
            print "Epoch: {} Cost Value: {}".format(epoch, cost)
        #TODO: add timestamp
        self.saver.save(self.sess, modelPath)
        print "Model was saved in the following path: " + modelPath

    # Use the model to predict the result of the xk uk input
    def predict(self, xk, uk):
        xk_uk_in = np.hstack((xk, uk))
        if self.restored is False:
            self.saver = tf.train.import_meta_graph(modelPath + '.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(modelDir))
            print "Model was restored from the following path: " + modelPath
            self.restored = True

        xOut = self.sess.run(self.xOut, feed_dict={self.xuIn: xk_uk_in})

        return xOut
