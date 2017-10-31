import os
import os.path
import argparse
import h5py
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
''' 
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )

args = parser.parse_args()
'''
var = load_h5py("dataset_partA.h5")
y = var[1]
x = var[0]
x = x/255.0
kfold = KFold(n_splits=5,shuffle = True)
kfold.get_n_splits(x,y)
s = 0
count = 0
savar = 0
for traini, testi in kfold.split(x):
	xtrain, xtest = x[traini],x[testi]
	ytrain, ytest = y[traini], y[testi]
	clf = MLPClassifier(hidden_layer_sizes=(100,50), activation='logistic',max_iter=100)
	n1,n2,n3 = xtrain.shape
	xtrain = xtrain.reshape(n1,n2*n3)
	clf.fit(xtrain,ytrain)
	print "fit"
	n1,n2,n3 = xtest.shape
	xtest = xtest.reshape(n1,n2*n3)
	#ypred = clf.predict(xtest)
	print "predict"
	s = s+clf.score(xtest,ytest)
	scr = clf.score(xtest,ytest)
	if scr > savar:
		joblib.dump(clf,'/home/tushita/Desktop/2.pkl')
	count = count+1
	print s

print "final"
print (s/count)*100
