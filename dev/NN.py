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
from sentiment import parse_1, date_transform, senti
from sklearn.metrics import accuracy_score

####################need x_train, y_train, x_test, y_test, traintest_X##################
def NN(df,fir,sec):
	traintest=df.loc[:,['1','2','3','4','polarity']]
	y=df.loc[:,'y']
	train_size=int(0.8*traintest.shape[0])
	x_train=traintest.iloc[-train_size:,:]
	x_test=traintest.iloc[:(traintest.shape[0]-train_size),:]
	y_train=y.iloc[-train_size:]
	y_test=y.iloc[:(traintest.shape[0]-train_size)]
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
		model.add(Dense(fir, input_dim = train_X.shape[1], init = 'he_normal', activation='tanh'))
		model.add(BatchNormalization())
		model.add(PReLU())  
		model.add(Dense(sec, init = 'he_normal', activation='tanh'))
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
	nfolds = 1
	print "model created then fold"
	
	testset = test_X
	ytestset = y_test

	print "-"*100
	print "KFold passed"
	print "-"*100

	## train models
	nbags = 5

	from time import time
	import datetime

	pred_test = np.zeros([testset.shape[0],1])
	begintime = time()
	count = 0
	filepath="weights.best.hdf5"
	print "-"*100
	print "Start train"
	print "-"*100
	pred_train=np.zeros([x_train.shape[0],1])
	for j in range(nbags):
		print(j)
		model = nn_model()
		model.fit(train_X, train_y, nb_epoch = 1200, batch_size=100, verbose = 0)
		pred_test += model.predict(x=testset, verbose=0)
		print "pred_test dimension is" , pred_test.shape
		print "testset dimension is" , testset.shape
		pred_train += model.predict(x=train_X,verbose=0)
		print "pred_train dimension is" , pred_train.shape
		print "train_X dimension is" , train_X.shape
		print(str(datetime.timedelta(seconds=time()-begintime)))
	pred_test=pred_test/nbags
	pred_train=pred_train/nbags
	pred_test=map(lambda x: 1 if x>0.5 else 0, pred_test)
	pred_train=map(lambda x: 1 if x>0.5 else 0, pred_train)

	print "train accuracy is " , accuracy_score(y_train,pred_train)
	print "test accuracy is ", accuracy_score(y_test, pred_test)
	return abs(accuracy_score(y_train,pred_train)-accuracy_score(y_test, pred_test))


def traintest():
	df=senti()
	df.drop('content',axis=1,inplace=True)
	market_data=pd.read_csv('table.csv',header=0)
	df_1=market_data.merge(df,how='right',left_on='Date',right_on='date')
	df_1.drop('date',axis=1,inplace=True)
	df_1=df_1.fillna(0)
	df_1.y=map(lambda x:1 if x>0 else 0,df_1.y)
	print "dataset created"

	NN(df_1,300,50)
	#map_result={}
	#first_hidden=range(10,310,10)
	#second_hidden=range(10,310,10)
	#for fir in first_hidden:
		#for sec in second_hidden:
			#map_result[(fir,sec)]=NN(df_1,fir,sec)



traintest()

