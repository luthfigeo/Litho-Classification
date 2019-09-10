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


np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#DataLoading
files = listdir("Dataset_text")
df = []
for file in files:
    print(file)
    df.append(pd.read_csv("./Dataset_text/"+file, sep='\t'))
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

db = db[['GR','NEUTPHI','DENS','DEEPRES','LITHOLOGY']]#,'GR_COR','NEUTPHI_COR','DENS_COR','DEEPRES_COR']]
db.to_csv('Dataset.csv', sep=',')

# DataSummary
def DatasetSummary(dataset):
        print(dataset.shape)
        print(dataset.describe())
DatasetSummary(db)

# DataVisualization
def DatasetGraph(dataset):
    sns.set()
    sns.pairplot(dataset,hue='LITHOLOGY',palette=sns.diverging_palette(220, 20, n=7))
    '''ax = scatter_matrix(dataset,figsize=(20,20), diagonal='kde')
    ax[3,0].set_xlim(0,300)
    ax[3,1].set_xlim(0,1)
    ax[3,2].set_xlim(1.7,2.7)
    ax[3,3].set_xlim(0.1,1000)
    ax[3,3].set_xscale('log')
    plt.savefig('data.jpg',dpi=100)
    plt.show()'''
#DatasetGraph(db)

# DataPreprocessing
db = db.dropna()

db['DENS'] = db['DENS'].apply(lambda x: (x / 1000) if x > 100 else x)
db['NEUTPHI'] = db['NEUTPHI'].apply(lambda x: (x / 100) if x > 1 else x)
db = db.replace(('Tuff','Lithic Tuff','Vitric Tuff','Lapili Tuff','Welded Tuff','Welded Lapili Tuff','Tuff Slate'),'Pyroclast')
db = db.replace(('Tuffaceous Sandstone','Coal','Limestone','Calcareous Shale','Mudstone','Shale','Shale sand'),'Sedimentary')
db = db.replace(('Andesite','Basalt','Slate'),'Igneous')

X = db.iloc[:,0:4].values
Y = db.iloc[:,-1].values
Y = Y.reshape((len(Y),1))

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
#encoder = LabelEncoder()
Y = ct.fit_transform(Y)

test = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test)'''
        
# Build models
def ModelKnn(xtrain,xtest,ytrain):
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(xtrain, ytrain)
        ypredict = classifier.predict(xtest)
        return ypredict, classifier

def ModelSvm(xtrain,xtest,ytrain):
        classifier = SVC(kernel='rbf', random_state=seed)
        classifier.fit(xtrain, ytrain)
        ypredict = classifier.predict(xtest)
        return ypredict, classifier

def ModelRF(xtrain,xtest,ytrain):
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
        classifier.fit(xtrain, ytrain)
        ypredict = classifier.predict(xtest)
        return ypredict, classifier

def ModelNBayes(xtrain,xtest,ytrain):
        classifier = GaussianNB()
        classifier.fit(xtrain, ytrain)
        ypredict = classifier.predict(xtest)
        return ypredict, classifier
    
def ModelAnn(xtrain, xtest, ytrain):
        classifier = Sequential()
        classifier.add(Dense(activation = 'relu', input_dim=4, init='uniform', output_dim=10))
        classifier.add(Dense(activation = 'softmax', init='uniform', output_dim=3))
        classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        classifier.fit(xtrain, ytrain, batch_size=10, nb_epoch=30)
        ypredict = classifier.predict(xtest)
        return ypredict, classifier

Y_predict, classifier = ModelRF(X_train,X_test,Y_train)

#Calculating Accuracy
cmARF = confusion_matrix(Y_test, Y_predict)
cvRF = model_selection.cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)    
accRF = accuracy_score(Y_test, Y_predict)

#Out
test_predict = pd.DataFrame(Y_test, Y_predict)
test_predict.to_csv('Test_Predict17.csv', sep=',')

#BlindPrediction
def BlindPrediction(file,classifier):
    db = pd.read_csv("./blindtest_test/"+file, sep='\t')
    db['DEEPRES'] = pd.to_numeric(db[dres_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['NEUTPHI'] = pd.to_numeric(db[nphi_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DENS'] = pd.to_numeric(db[rhob_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DEEPRES_COR'] = pd.to_numeric(db[dres_cor_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DENS_COR'] = pd.to_numeric(db[rhob_cor_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['NEUTPHI_COR'] = db['NPHI_COR']
    
    db = db[['WELL','DEPTH','GR','NEUTPHI','DENS','DEEPRES','LITHOLOGY']]
    db = db.dropna()
    
    db['DENS'] = db['DENS'].apply(lambda x: (x / 1000) if x > 100 else x)
    db['NEUTPHI'] = db['NEUTPHI'].apply(lambda x: (x / 100) if x > 1 else x)
    X = db.iloc[:,2:6].values
    Y_predict = classifier.predict(X)