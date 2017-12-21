#!/usr/bin/python3
#First Titanic python program

import tensorflow as tf
import numpy
import csv
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)
import shutil
import os
import re

SHUFFLE=False
INPUTSTEPS=None
EPOCHS=200
HIDDENUNITS=[500,500]

class Titanic:

	#FEATURES = ["Pclass","Sex","Age","SibSp","Parch"]
	#FEATURES = ["Pclass","Sex","Age","SibSp","Parch","companions"]
	#FEATURES = ["Pclass","Sex","Age","companions"] #Test Accuracy: 0.821229
	FEATURES = ["Pclass","Sex","Age","SibSp","Parch","Title"] #Test Accuracy: 0.854749 1x with shuffle... 
	#FEATURES = ["Pclass","SibSp","Parch"]
	LABEL = ["Survived"]


	def __init__(self, csvFilename):
		self.columnMap={}
		#Clear model to start from scratch every time!
		if os.path.exists('tf_model'):
			shutil.rmtree('tf_model')

		self.resultFile=open("results.txt","w")
		self.resultFile.write("FEATURES: "+str(self.FEATURES)+"\nSHUFFLE: "+str(SHUFFLE)+"\nINPUTSTEPS: "+str(INPUTSTEPS)+"\nEPOCHS: "+str(EPOCHS)+"\nHIDDENUNITS: "+str(HIDDENUNITS)+"\n\n")
		self.resultFile.flush()

		self.csvFilename=csvFilename
		print("Starting Titanic machine learning exercise")

	def __del__(self):
		print("Model Finished")

	def printResults(self):
		self.resultFile.close()
		#Display results on STDOUT
		with open("results.txt","r") as file:
			for line in file:
				print(line)
			file.close()

	def featureEngineering(self, fullDict):
		#Create title feature
		fullDict["Title"]=[]
		for i in range(0,len(fullDict["Name"])):
			fullDict["Title"].append(re.sub("^[^,][^,]*[,][ ]([^ ][^ ]*) .*$","\\1",fullDict["Name"][i]))

		#Create Age buckets feature
		fullDict["AgeBuckets"]=[]
		for i in range(0,len(fullDict["Age"])):
			if fullDict["Age"][i]== None or fullDict["Age"][i]=="":
				fullDict["Age"][i]="-1"
				fullDict["AgeBuckets"].append("99_Unknown")
			else:
				if float(fullDict["Age"][i]) <= 2:
					fullDict["AgeBuckets"].append("Baby")
				elif float(fullDict["Age"][i])<=10:
					fullDict["AgeBuckets"].append("Child")
				elif float(fullDict["Age"][i])<20:
					fullDict["AgeBuckets"].append("Teen")
				elif float(fullDict["Age"][i])<30:
					fullDict["AgeBuckets"].append("Twenties")
				elif float(fullDict["Age"][i])<40:
					fullDict["AgeBuckets"].append("Thirties")
				elif float(fullDict["Age"][i])<60:
					fullDict["AgeBuckets"].append("FortiesFifties")
				else:
					fullDict["AgeBuckets"].append("Elderly")

		#Create companion
		fullDict["companions"]=[]
		for i in range(0,len(fullDict["Parch"])):
			fullDict["companions"].append(float(fullDict["Parch"][i])+float(fullDict["SibSp"][i]))

		return fullDict

	def resolveFloat(self, k, val):
		try:
			return float(val)
		except ValueError:
			if k in self.columnMap:
				if val not in self.columnMap[k]:
					self.columnMap[k]["num_idx"]=self.columnMap[k]["num_idx"]+1.0
					self.columnMap[k][val]=self.columnMap[k]["num_idx"]
			else:
				self.columnMap[k]={}
				self.columnMap[k]["num_idx"]=0.0
				self.columnMap[k][val]=0.0
			return self.columnMap[k][val]

	def partitionTestSet(self,fullDict,sampleSegment,modulo):
		train_set={}
		test_set={}
		testSegment=0
		if modulo > 0:
			testSegment=sampleSegment%modulo
		#print("Arbitary value length = "+str(len(next (iter (fullDict.values())))))
		for n in range(0,len(fullDict[self.FEATURES[0]])):
			for k in fullDict:
				if k not in train_set:
					train_set[k]=[]
					test_set[k]=[]
				val = self.resolveFloat(k,fullDict[k][n])
				if modulo > 0 and n%modulo == testSegment:
					test_set[k].append(val)
				else:
					train_set[k].append(val)
		return train_set, test_set

	#https://www.tensorflow.org/get_started/input_fn
	def get_input_fn(self, data_set, num_epochs=1, shuffle=False):
		numpyDict = {}
		for k in self.FEATURES:
			#print(str(k)+" first 10 items "+str(data_set[k][0:10]))
			numpyDict[k]=numpy.asarray([float(i) for i in data_set[k]])
		targetArray=None
		if self.LABEL[0] in data_set:
			targetArray=numpy.asarray([float(i) for i in data_set[self.LABEL[0]]])
		return tf.estimator.inputs.numpy_input_fn(
			x=numpyDict,
			y=targetArray,
			num_epochs=num_epochs,
			shuffle=shuffle)

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
		print(str(self.train_set))

	def repopulateModel(self,sampleSegment,modulo):

		fullDict=self.loadCsvData(self.csvFilename)
		fullDict=self.featureEngineering(fullDict)
		self.train_set, self.test_set=self.partitionTestSet(fullDict,sampleSegment,modulo)
		print("train_set_size="+str(len(self.train_set[self.FEATURES[0]])))
		print("test_set_size="+str(len(self.test_set[self.FEATURES[0]])))

	def trainModel(self,sampleSegment,modulo):
		self.repopulateModel(sampleSegment,modulo)
		# Feature cols
		feature_cols = [tf.feature_column.numeric_column(k) for k in self.FEATURES]
		#tf.estimator.
		# Build 1 layer fully connected DNN with 10, 10 units respectively.
		self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols,
			n_classes=2,
			optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
			hidden_units=HIDDENUNITS,
			model_dir="tf_model")

		# Train
		self.classifier.train(input_fn=self.get_input_fn(self.train_set, shuffle=SHUFFLE, num_epochs=EPOCHS), steps=INPUTSTEPS)
		#steps=50   loss final = 78.835
		#steps=500  loss final = 80.6114
		#steps=5000 loss final = 76.5161
		#steps=50000loss final = 68.7828

	def evaluateModel(self,sampleSegment,modulo):
		self.repopulateModel(sampleSegment,modulo)
		# Evaluate accuracy.
		print("Evaluate accuracy score")
		accuracy_score = self.classifier.evaluate(input_fn=self.get_input_fn(self.test_set))["accuracy"]

		self.resultFile.write("Test Accuracy: {0:f}\n".format(accuracy_score)+"\n")
		self.resultFile.flush()

	def predictCsv(self,testCsv):
		fullDict=self.loadCsvData(testCsv)
		fullDict=self.featureEngineering(fullDict)
		fullDict, testDict = self.partitionTestSet(fullDict,0,0)

		#predictions = list(self.classifier.predict(input_fn=self.get_input_fn(fullDict),outputs="Survived"))
		#predicted_classes = [p["classes"] for p in predictions]
		#print("New Samples, Class Predictions:    {}\n".format(predicted_classes))

		cnt=0
		for p in self.classifier.predict(input_fn=self.get_input_fn(fullDict),predict_keys=["classes"]):
			print("prediction="+str(p))
		print("prediction count = "+str(cnt))

titanic = Titanic("../data/train.csv")
#titanic.printCsvDict()
#print(str(titanic.get_input_fn(titanic.train_set)))
titanic.trainModel(0,5)
#titanic.trainModel(1,5)
#titanic.trainModel(2,5)
#titanic.trainModel(3,5)
#titanic.trainModel(4,5)
titanic.evaluateModel(0,5)
#titanic.evaluateModel(1,5)
#titanic.evaluateModel(2,5)
#titanic.evaluateModel(3,5)
#titanic.evaluateModel(4,5)
titanic.predictCsv("../data/test.csv")
#titanic.printResults()