from keras.models import Sequential,load_model
from keras.layers import Dense
from keras import optimizers
import numpy as np

class Emulator():
    def __init__(self, new=False,filePath=''):
        if new:
            self.model = Sequential()
            self.model.add(Dense(100, input_dim=10, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(75, input_dim=100, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(50, input_dim=75, kernel_initializer='normal', activation='tanh'))
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
        self.model.fit(input, target, batch_size=100, epochs=100, verbose=0,validation_split=0.3)
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