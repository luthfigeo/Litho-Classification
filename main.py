import sys
import scipy
import numpy as np
import matplotlib
import sklearn
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
def LoadDataset{
        address = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        attr = ['GR', 'NPHI', 'RHOB', 'ILD', 'lithology']
        dataset = pd.read_csv(address, names=attr)
        }

# Data Summary
def DatasetSummary{
        print(dataset.shape)
        print(dataset.describe())
        
        }
# Data visualization
def DatasetGraph{
        scatter_matrix(dataset)
        plt.show()
        }

# Validation
def DatasetValidation{
        array = dataset.values
        X = array[:,0:4]
        Y = array[:,4]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        }

# Build models
def ModelLinear{
        
        }

def ModelTree{
        
        }

def ModelKnn{
        for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
        trainingSet=[]
        testSet=[]
        loadDataset('Jatibarang Fm', 0.7, trainingSet, testSet)
        print 'Train: ' + repr(len(trainingSet))
        print 'Test: ' + repr(len(testSet))
        }

def ModelNbayes{
        
        }

def ModelSvm{
        
        }

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main{
        trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
        }