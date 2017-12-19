#!/usr/bin/python3
#First Titanic python program

import tensorflow as tf
import csv

class Titanic:
	self.FEATURES = ["Pclass","Sex","Age","SibSp","Parch"]
	self.LABEL = ["Survived"]

	def __init__(self, csvFilename):
		print("Starting Titanic machine learning exercise")
		self.loadCsvData(csvFilename)

	def loadCsvData(self, csvFilename):
		self.columnDict={}
		self.dataDict={}
		with open(csvFilename) as cf:
			dataReader = csv.reader(cf, delimiter=',', quotechar='"')
			rowNum = 1
			for row in dataReader:
				colNum=0
				for colval in row:
					if rowNum == 1:
						self.columnDict[colNum]=colval
						self.dataDict[colval]=[]
					else:
						self.dataDict[self.columnDict[colNum]].append(colval)
					colNum=colNum+1
				rowNum=rowNum+1

	def printCsvDict(self):
		print(str(self.dataDict))


titanic = Titanic("../data/train.csv")
titanic.printCsvDict()