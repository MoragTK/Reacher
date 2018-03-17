import time
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from DataClass import Rand_data

class Kann:

    model = Sequential()
    model.add(Dense(25, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(14, input_dim=25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    def random_train(self,sample=1000,cycle=1,batch_size=10,epochs=1000,verbose=0):
        for i in range(1,cycle):
            data = Rand_data()
            data.make_data(sample)
            d = data.get_data()
            X = d[0]
            Y = d[1]
            self.model.fit(X, Y, batch_size,epochs,verbose)
            name=time.gmtime()
            self.model.save("/home/omer/workspace/model")


    seed = 7
    numpy.random.seed(seed)
    # evaluate model with standardized dataset



    print(model.predict(X[0]))
    '''''
   
    estimator = KerasRegressor(model, nb_epoch=100, batch_size=5, verbose=2)
    
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    
    self, x, batch_size=None, verbose=0, steps=None 
    
    
    print (build_fn.predict)
    
    '''