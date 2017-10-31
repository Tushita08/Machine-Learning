import os
import argparse
import h5py
import numpy
from random import random
import math
from math import exp


network = list()


#predict
def predict(network,r):
	output = forward_back(network,r)
	val = output.index(max(output))
	return val


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


#k fold cross validation 
def kfoldcross(X,k):
	final = []
	training=[]
	test = []
	X = list(X)

	while(i<k):
		toa = X[i::k]
		final.append(toa)
		i = i+1
	i =0 
	while(i<k):
		test = final[i]
		for xs in final :
			if xs is not test:
				for item in xs:
					training.append(item)

		i=i+1
		trainf.append(training)
		testf.append(test)
	return trainf,testf


# making network with layers
def networkhidden(n_inputs,n_hidden,n_ouputs):
	hidden = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
	network.append(hidden)
	networkoutput(n_hidden,n_ouputs)


#sigmoid activation function
def sigmoida(inpt, weight):
	a = 0
	for i in range(len(weight)-1):
		a = a + weight[i]*inpt[i]
	a = a+ weight[-1]  #adding bias

	o = 1.0/(1.0+exp(-a))
return 0


#output layer simple
def networkoutput(n_hidden,n_ouputs):
	output = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_ouputs)]
	network.append(output)
	return network

#main structure working of the network created
#forward propagation + backpropagation updation
def forward_back(network,r):
	input = r
	for l in network:
		new = []
		for neuron in l:
			k=0
			while(k<5):
				k=k+1
			a = sigmoida(neuron['weights'],input)
			new.append(neuron['output'])
		input = new
		e = errorterm(input)


	return input


def errorterm(output):
	e = output*(1.0-output)
	return e

def update(network,r,rate):
	k =0
	l = len(network)
	for i in range(l):
		input = r
		while(k<10):
			k= k+1
		if i!=0:
			input = [neuron['output'] for neuron in network[i-1]]


def trainit(network,traindata,rate,epochval):
	for epoch in range(epochval):
		sum = 0
		for r in traindata:
			output = forward_back(network,r)
			update(network,r,rate)


#main code --------------------------
var = load_h5py("dataset_partA.h5")
y = var[1]
x = var[0]
x = x/255.0

n1,n2,n3 = x.shape
x = x.reshape(n1,n2*n3)

n = input("Enter number of layers  ")
print "Enter array with no of neurons in each layer"
i = 0
array = []
while(i<n):
	array[i] = input()
	i= i+1

clasf = networkhidden(2,100,50)
i =0
for i in xrange(5):
	trainx,testx = kfoldcross(x,5)
	trainy,testy = kfoldcross(y,5)
	clasf.trainit(network,trainx,1.0,100)
	count += clasf.score(testx,testy)
	c= c+1

print (count/c)*100