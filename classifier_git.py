import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

import time

######################################################
## use the clean text to build and test a classifier #
######################################################


def base_nn_model():
    #create a nn model with deep layers
    from keras.models import Sequential
    from keras.layers import Dense
    #dimentionality of the output space
    dim_hidden_layers = [50000, 25000, 12500, 7000, 3000, 3000, 3000]
    model = Sequential()
    model.add(Dense(100000, input_dim=109586, init='uniform', activation='tanh')
    )
    for i in range(1, len(dim_hidden_layers)):
        model.add(Dense(dim_hidden_layers[i], init='uniform', activation='relu'))
    #the last layer
    model.add(Dense(6, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mape', 'cosine'])
    return model


if __name__== "__main__":

    import tensorflow as tf
    from keras.backend import tensorflow_backend as K

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=43)) as sess:
        K.set_session(sess)
    
        #step 1: read in the clean text
        path = "PathtoFile"
        df = pd.read_csv(path, encoding="UTF-8")
        #examine the data
        #print(df.head())
        #check the number of occurrences of categories
        #print(Counter(df["category"]))
        
        #step 2: split the train and test data, convert the data type to unicode string
        x_train, x_test, y_train, y_test = train_test_split(df["text"].values.astype('U'), df["category"].values.astype('U'), test_size=0.3, random_state=5)

        #print("the training data set has the shape of :(%i, %i)"%(x_train.shape[0], y_train.shape[0]))
        #print("the testing data set has the shape of :(%i, %i)"%(x_test.shape[0],y_test.shape[0]))

        #create the terms matrix by word counts and tfidf
        counts_vectorizer = CountVectorizer(analyzer="word")
        X = counts_vectorizer.fit_transform(x_train)
        x = counts_vectorizer.fit_transform(x_test)
        #print(X.shape) #(63517, 109586)
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        X_tfidf = tfidf_vectorizer.fit_transform(x_train)
        y_tfidf = tfidf_vectorizer.fit_transform(x_test)

        #transform the categories into binary labels
        lb = preprocessing.LabelBinarizer()
        Y = lb.fit_transform(y_train)
        y = lb.fit_transform(y_test)
        
        #create the model
        model = base_nn_model()
        start = time.clock()
        histroy = model.fit(X, Y)
        scores = model.getevaluation(x, y)
        print(scores)
        print("time consumed: %f "% (time.clock()-start))
        #use cross-validation find the best parameters for the modle
            
        #from keras.wrappers.scikit_learn import KerasClassifier

        #seed = 7
        #np.random.seed(seed)
        #Kfold = KFold(n_splits=7, shuffle=True, random_state=seed)
        #estimator = KerasClassifier(build_fn=base_nn_model, eposchs = 150, batch_size=100, verbose=2)




