import glob
import sys
import scipy
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
from os import listdir
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils import np_utils

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#DataLoading
def DatasetLoading(folder):
    files = listdir(folder)
    df = []
    db = []
    for file in files:
        print(file)
        df.append(pd.read_csv("./"+folder+"/"+file, sep='\t'))
    db = pd.concat(df, ignore_index=True, sort=True)
    dres_cols = ['HDRS', 'ILD', 'LLD', 'RLA5']
    nphi_cols = ['NPHI', 'TNPH']
    rhob_cols = ['RHOB', 'RHOZ']
    rhob_cor_cols = ['RHOB_COR', 'RHO_COR']
    dres_cor_cols = ['ILD_COR', 'LLD_COR']
    db['DEEPRES'] = pd.to_numeric(db[dres_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['NEUTPHI'] = pd.to_numeric(db[nphi_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DENS'] = pd.to_numeric(db[rhob_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DEEPRES_COR'] = pd.to_numeric(db[dres_cor_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DENS_COR'] = pd.to_numeric(db[rhob_cor_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['NEUTPHI_COR'] = db['NPHI_COR']
    
    db['DENS'] = db['DENS'].apply(lambda x: (x / 1000) if x > 100 else x)
    db['NEUTPHI'] = db['NEUTPHI'].apply(lambda x: (x / 100) if x > 1 else x)
    
    return db

db = DatasetLoading(folder="Dataset_text")
db = db[['GR','NEUTPHI','DENS','DEEPRES','LITHOLOGY']]#,'GR_COR','NEUTPHI_COR','DENS_COR','DEEPRES_COR']]
db.to_csv('Dataset1.csv', sep=',')

# DataSummary
def DatasetSummary(dataset):
        print(dataset.shape)
        print(dataset.describe())
DatasetSummary(dbtuff.where(db['DATATYPE']=='Coreplug',axis=1))

# DataVisualization
def DatasetGraph(dataset,category):
    sns.set()
    g = sns.pairplot(dataset.dropna(),hue=category)
    g.axes[0,0].set_ylim((0,300))
    g.axes[3,0].set_xlim((0,300))
    g.axes[1,0].set_ylim((0,1))
    g.axes[3,1].set_ylim((0,1))
    g.axes[3,2].set_xlim((1.7,2.7))
    g.axes[2,0].set_ylim((1.7,2.7))
    g.axes[3,3].set_xlim((0.1,1000))
    g.axes[3,0].set_ylim((0.1,1000))
    g.axes[3,0].set(yscale='log')
    g.axes[3,3].set(xscale='log')
    g.fig.suptitle("Dataset Distribution (mudlog)")
    g.savefig('mudlog_datadist.jpg',dpi=100)
DatasetGraph(db.where(db['DATATYPE']=='MudLog'),category='LITHOLOGY')

# DataPreprocessing
db = db.dropna()

#4 lithology
db = db.replace(('Tuff','Lithic Tuff','Vitric Tuff','Lapili Tuff','Welded Tuff','Welded Lapili Tuff','Tuff Slate'),'Pyroclast')
db = db.replace(('Tuffaceous Sandstone','Coal','Limestone','Calcareous Shale','Mudstone','Shale','Shale sand'),'Sedimentary')
db = db.replace(('Andesite','Basalt'),'Igneous')

X = db.iloc[:,0:3].values
Y = db.iloc[:,-1].values
Y = Y.reshape((len(Y),1))

test = 0.20
seed = 0
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)
Y_train = Y_train.reshape((len(Y_train),1))
Y_test = Y_test.reshape((len(Y_test),1))
Y_train_cat = np_utils.to_categorical(Y_train)
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
Y_train = ct.fit_transform(Y_train)

lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test)

        
# Build models
def ModelKnn():
        classifier = KNeighborsClassifier(n_neighbors=10)
        return classifier

def ModelSvm():
        classifier = SVC(kernel='rbf', random_state=seed)
        return classifier

def ModelRF():
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
        return classifier

def ModelNBayes():
        classifier = GaussianNB()
        return classifier
    
def ModelAnn():
        classifier = Sequential()
        classifier.add(Dense(activation = 'relu', input_dim=2, kernel_initializer='uniform', units=10))
        classifier.add(Dense(activation = 'softmax', kernel_initializer='uniform', units=17))
        classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier

classifier = ModelAnn()
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)

#Y_predict = Y_predict.apply(lambda x: x.idxmax(), axis = 1)

#Calculating Accuracy
cmSvmi_ = confusion_matrix(Y_test, Y_predict)
cvSvmi_ = model_selection.cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)    
accSvmi_ = accuracy_score(Y_test, Y_predict)

estimator = KerasClassifier(build_fn=ModelAnn, epochs=30, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
cvAnn4 = model_selection.cross_val_score(estimator, X=X_train, y=Y_train_cat, cv=kfold)

#Visualize
X_test = scaler.inverse_transform(X_test)
prediction = pd.DataFrame(X_test,columns=['GR', 'NEUTPHI','DENS'])
prediction['PREDICTION'] = Y_predict
DatasetGraph(prediction,'PREDICTION')
X_train = scaler.inverse_transform(X_train)
trainingset = pd.DataFrame(X_train,columns=['GR', 'NEUTPHI','DENS'])
trainingset['LITHOLOGY'] = Y_train
DatasetGraph(trainingset,'LITHOLOGY')

#Parameter Picking
parameters = [{'C':[1, 10], 'kernel':['linear']},
               {'C':[1, 10], 'kernel':['rbf'], 'gamma': [0.1, 0.2]}]
gridsearch = model_selection.GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
gridsearch = gridsearch.fit(X_train, Y_train)
best_accuracy = gridsearch.best_score_
best_parameters = gridsearch.best_params_

#Out
test_predict = pd.DataFrame(Y_test, Y_predict)
test_predict.to_csv('Test_Predict17.csv', sep=',')

#BlindPrediction
def BlindPrediction(file,classifier):
    db_blind = DatasetLoading(folder=file)
    db_blind = db_blind[['WELL','DEPTH','GR','NEUTPHI','DENS','DEEPRES']]
    db_blind = db_blind.dropna()
    X_blind = db_blind.iloc[:,2:6].values
    X_blind = scaler.transform(X_blind)
    Y_blind = classifier.predict(X_blind)
    
    return db_blind, Y_blind

data, prediction = BlindPrediction(file="blindtest/blindtest_text",classifier)
result = pd.DataFrame(data, prediction)
result.to_csv('Result.csv', sep=',')