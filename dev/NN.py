import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import zscore
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sentiment import parse, date_transform

####################need x_train, y_train, x_test, y_test, traintest_X##################
def NN():
	train_X = x_train.as_matrix()
	test_X = x_test.as_matrix()
	print "before scalar train size",train_X.shape
	print "before scalar test size",test_X.shape

	traintest = np.vstack((train_X, test_X))

	traintest = preprocessing.StandardScaler().fit_transform(traintest)

	train_X = traintest[range(train_X.shape[0])]
	test_X = traintest[range(train_X.shape[0], traintest.shape[0])]
	print "after scalar train size",train_X.shape
	print "after scalar test size", test_X.shape
	## neural net
	def nn_model():
		model = Sequential()   
		model.add(Dense(300, input_dim = train_X.shape[1], init = 'he_normal', activation='relu'))
		model.add(BatchNormalization())
		model.add(PReLU())  
		model.add(Dense(50, init = 'he_normal', activation='sigmoid'))
		model.add(BatchNormalization())    
		model.add(PReLU())	
		model.add(Dense(1, init = 'he_normal', activation='sigmoid'))
		model.compile(loss = 'binary_crossentropy', optimizer = 'adam')#, metrics=['accuracy'])
		return(model)
	train_y = y_train
	print "this is train Y",train_y
	print '-'*50
	do_all = True
	## cv-folds
	nfolds = 10
	if do_all:
		if nfolds>1:
			folds = KFold(int(len(train_y)), n_folds = nfolds, shuffle = True, random_state = 111)
		pred_oob = np.zeros(len(train_y))
		testset = test_X
	else:
		folds = KFold(int(len(train_y)*0.8), n_folds = nfolds, shuffle = True, random_state = 111)
		pred_oob = np.zeros(int(len(train_y)*0.8))
		testset = train_X[range(int(len(train_y)*0.8), len(train_y))]
		ytestset = train_y[int(len(train_y)*0.8):(len(train_y))]

	print "-"*100
	print "KFold passed"
	print "-"*100

	## train models
	nbags = 1

	from time import time
	import datetime

	pred_test = np.zeros(testset.shape[0])
	begintime = time()
	count = 0
	filepath="weights.best.hdf5"
	print "-"*100
	print "Start train"
	print "-"*100
	if nfolds>1:
		for (inTr, inTe) in folds:
			count += 1
			print "INTR&INTE read"
			xtr = train_X[inTr]
			ytr = train_y[inTr]
			xte = train_X[inTe]
			yte = train_y[inTe]
			pred = np.zeros(xte.shape[0])
			print(0)
			print "Model created"
			model = nn_model()
			early_stop = EarlyStopping(monitor='val_loss', patience=75, verbose=0)
			checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
			print "model fit"
			model.fit(xtr, ytr, nb_epoch = 1200, batch_size=100, verbose = 0, validation_data=[xte, yte])

			pred += model.predict(x=xte, verbose=0)
		        
			pred_test += model.predict(x=testset, verbose=0)
		        
			print(log_loss(yte,pred/(1)))
			if  not do_all:
				print(log_loss(ytestset,pred_test/(1+count*nbags)))
				print("accuracy on test is", (ytestset==pred_test/(nbags * count))/len(ytestset))
			print(str(datetime.timedelta(seconds=time()-begintime)))
		pred /= nbags
		pred_oob= pred
		score = log_loss(yte,pred)
		print('Fold ', count, '- logloss:', score)
		if not do_all:
			print(log_loss(ytestset, pred_test/(nbags * count)))
			print("accuracy on test is", (ytestset==pred_test/(nbags * count))/len(ytestset))
	else:
		for j in range(nbags):
			print(j)
			model = nn_model()
			model.fit(train_X, train_y, nb_epoch = 1200, batch_size=100, verbose = 0)
			pred_test += model.predict_proba(x=testset, verbose=0)
			print(str(datetime.timedelta(seconds=time()-begintime)))

	if nfolds>1:
		if do_all:
			print('Total - logloss:', log_loss(train_y, pred_oob))
			print("accuracy on train is", (train_y==pred_oob)/len(train_y))
		else:
			print('Total - logloss:', log_loss(train_y[0:int(len(train_y)*0.8)], pred_oob))


def traintest():
	

