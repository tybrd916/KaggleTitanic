#!/usr/bin/python3
#First Titanic python program

import tensorflow as tf
import numpy
import csv

class Titanic:
	FEATURES = ["Pclass","Sex","Age","SibSp","Parch"]
	LABEL = ["Survived"]


	def __init__(self, csvFilename):
		print("Starting Titanic machine learning exercise")
		fullDict=self.loadCsvData(csvFilename)
		self.train_set, self.test_set=self.partitionTestSet(fullDict,5)
		print("train_set_size="+str(len(self.train_set[self.FEATURES[0]])))
		print("test_set_size="+str(len(self.test_set[self.FEATURES[0]])))

	def partitionTestSet(self,fullDict,modulo):
		train_set={}
		test_set={}
		print("Arbitary value length = "+str(len(next (iter (fullDict.values())))))
		for n in range(0,len(fullDict[self.FEATURES[0]])):
			for k in fullDict:
				if k not in train_set:
					train_set[k]=[]
					test_set[k]=[]
				if n%modulo == 0:
					test_set[k].append(fullDict[k][n])
				else:
					train_set[k].append(fullDict[k][n])
		return train_set, test_set


	#https://www.tensorflow.org/get_started/input_fn
	def get_input_fn(self, data_set, num_epochs=None, shuffle=True):
		numpyDict = {}
		for k in self.FEATURES:
			numpyDict[k]=numpy.asarray(data_set[k])
		return tf.estimator.inputs.numpy_input_fn(
    		x={"x": numpyDict},
    		y=numpy.asarray(data_set[self.LABEL[0]]),
    		num_epochs=None,
    		shuffle=True)

	def loadCsvData(self, csvFilename):
		columnDict={}
		dataDict={}
		with open(csvFilename) as cf:
			dataReader = csv.reader(cf, delimiter=',', quotechar='"')
			rowNum = 1
			for row in dataReader:
				colNum=0
				for colval in row:
					if rowNum == 1:
						columnDict[colNum]=colval
						dataDict[colval]=[]
					else:
						dataDict[columnDict[colNum]].append(colval)
					colNum=colNum+1
				rowNum=rowNum+1

		return dataDict

	def printCsvDict(self):
		print(str(self.dataDict))

	#def trainModel(self):



titanic = Titanic("../data/train.csv")
#titanic.printCsvDict()
print(str(titanic.get_input_fn(titanic.train_set)))