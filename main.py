import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir

from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras.optimizers import SGD

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
        df.append(pd.read_csv("./"+folder+"/"+file, sep='\t', error_bad_lines=False))
    db = pd.concat(df, ignore_index=True, sort=True)
    #dres_cols = ['ILD','LLD','RLA5']
    #nphi_cols = ['NPHI','TNPH']
    #rhob_cols = ['RHOB', 'RHOZ']
    dres_cols = ['ILD','LLD','RLA5']
    nphi_cols = ['NPHI','TNPH']
    rhob_cols = ['RHOB', 'RHOZ']
    db['DEEPRES'] = pd.to_numeric(db[dres_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['NEUTPHI'] = pd.to_numeric(db[nphi_cols].bfill(axis=1).iloc[:, 0], errors='coerce')
    db['DENS'] = pd.to_numeric(db[rhob_cols].bfill(axis=1).iloc[:, 0], errors='coerce')

    db['DENS'] = db['DENS'].apply(lambda x: (x / 1000) if x > 100 else x)
    db['NEUTPHI'] = db['NEUTPHI'].apply(lambda x: (x / 100) if x > 1 else x)

    return db

db = DatasetLoading(folder="Dataset_text")
db = db[['GR','NEUTPHI','DENS','DEEPRES','LITHOLOGY']]#,'GR_COR','NEUTPHI_COR','DENS_COR','DEEPRES_COR']]
db = db[['GR_COR','NPHI_COR','RHOB_COR','ILD_COR','LITHOLOGY']]#,'GR_COR','NEUTPHI_COR','DENS_COR','DEEPRES_COR']]

# DataSummary
def DatasetSummary(dataset):
        print(dataset.shape)
        print(dataset.describe())
DatasetSummary(db)

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
    #g.fig.suptitle("Lithology Dis")
    g.savefig('data12.jpg',dpi=100)
DatasetGraph(db,category='LITHOLOGY')

litho_colors = ['#b14801', '#ffc1b7', '#ddb397', '#56e0fc', '#820041', '#f9b5bb',
       '#f9d3d3', '#43aff9', '#f9d3d3', '#cfefdf', '#ace4c8',
        '#cdffd9', '#e6cdff', '#f9d3d3','#f9d3d3',  '#cdffd9',
       '#f9d3d3', '#fff3c9','#fff3c9','#cdffd9']

litho_counts = db['LITHOLOGY'].value_counts().sort_index()
litho_counts.plot(kind='bar', color = litho_colors, title='Distribution of Data by Lithology')

# DataPreprocessing
db = db.dropna()

#4 lithology
db = db.replace(('Tuff','Lithic Tuff','Vitric Tuff','Lapili Tuff','Welded Tuff','Welded Lapili Tuff'),'Tuff')
db = db.replace(('Tuffaceous Sandstone','Coal','Limestone','Calcareous Shale','Mudstone','Shale','Shale sand'),'Sedimentary')
db = db.replace(('Andesite','Basalt'),'Igneous')
db = db[db.LITHOLOGY != 'Shale sand']
db = db[db.LITHOLOGY != 'Slate']
db = db[db.LITHOLOGY != 'Tuff Slate']
db = db[db.LITHOLOGY != 'Coal']
db = db.replace(('Mudstone','Shale'),'Shale')
db = db.replace(('Lithic Tuff','Lapili Tuff','Welded Tuff','Welded Lapili Tuff'),'Lithic Tuff')
db = db.replace(('Vitric Tuff','Tuff'),'Vitric Tuff')

X = db.iloc[:,0:4].values
Y = db.iloc[:,-1].values
Y = Y.reshape((len(Y),1))

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

Y = label_binarize(Y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
litho_label = ['Andesite','Basalt','Conglomerate','Lapili Tuff','Lithic Tuff',
               'Mudstone','Shale','Tuff','Tuffaceous Sandstone','Vitric Tuff', 
               'Welded Lapili Tuff', 'Welded Tuff']
n_classes = Y.shape[1]

test = 0.20
seed = 0
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# lda = LDA(n_components=2)
# X_train = lda.fit_transform(X_train,Y_train)
# X_test = lda.transform(X_test)
# =============================================================================

# Build models
def ModelKnn():
        classifier = KNeighborsClassifier(n_neighbors=3)
        return classifier

def ModelSvm():
        classifier = SVC(kernel='rbf', random_state=seed)
        return classifier

def ModelRF():
        classifier = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
        return classifier
    
def ModelBRF():
        classifier = BalancedRandomForestClassifier(n_estimators=200, random_state=seed)
        return classifier

def ModelNBayes():
        classifier = GaussianNB()
        return classifier
    
def ModelAnn():
        classifier = Sequential()
        classifier.add(Dense(activation = 'relu', input_dim=4, kernel_initializer='uniform', units=10))
        classifier.add(Dense(activation = 'softmax', kernel_initializer='uniform', units=17))
        lr = 0.1
        decay = 0.0001
        momentum = 0.75
        sgd= SGD(lr=lr, momentum=momentum, decay=decay,nesterov=False)
        classifier.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])
        classifier = KerasClassifier(build_fn=ModelAnn, epochs=100, batch_size=5, verbose=0)
        return classifier
    
classifier = ModelAnn()
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)
Y_pred = Y_predict.argmax(1)
Y_tes = Y_test.argmax(1)

#Calculating Accuracy
cm = confusion_matrix(Y_tes, Y_pred)
cv = model_selection.cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
acc = accuracy_score(Y_test, Y_predict)
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
#Fmeasure
F1 = 2*PPV*TPR/(PPV+TPR)

performance = pd.DataFrame([TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC,F1])
performance.index = (['TPR','TNR','PPV','NPV','FPR','FNR','FDR','ACC','F1'])

fpr_auc = dict()
tpr_auc = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr_auc[i], tpr_auc[i], treshold = roc_curve(Y_test[:,i], Y_predict[:,i])
    roc_auc[i] = auc(fpr_auc[i], tpr_auc[i])
    
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr_auc[i], tpr_auc[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic %s' %litho_label[i])
    plt.legend(loc="lower right")
    plt.show()

#ANN


# =============================================================================
# #Parameter Picking
# parameters = [{'C':[1, 10], 'kernel':['linear']},
#                {'C':[1, 10], 'kernel':['rbf'], 'gamma': [0.1, 0.2]}]
# gridsearch = model_selection.GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
# gridsearch = gridsearch.fit(X_train, Y_train)
# best_accuracy = gridsearch.best_score_
# best_parameters = gridsearch.best_params_
# =============================================================================

#BlindPrediction
def BlindPrediction(file,classifier):
    db_blind = DatasetLoading(folder=file)
    db_blind = db_blind[['WELL','DEPTH','GR','NPHI','RHOB','ILD']]
    db_blind = db_blind.dropna()
    X_blind = db_blind.iloc[:,2:6].values
    X_blind = scaler.transform(X_blind)
    Y_blind = classifier.predict(X_blind)
    return db_blind, Y_blind

data, prediction = BlindPrediction(file="blindtest/blindtest_text",classifier=classifier)
prediction = prediction.argmax(1)
data['LITHOLOGY2'] = prediction
prediction = encoder.inverse_transform(prediction)
data['LITHOLOGY'] = prediction
data.to_csv('Result_KNN3_12_1410.csv', sep=',')