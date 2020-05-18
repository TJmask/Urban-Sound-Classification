##importing basic packages
import glob
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import IPython.display as ipd

## importing model pacakage
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical






##################################################################################################
############################## data and feature preprocessing ###################################
##################################################################################################

# feature extracting includes MFCC, Melspectrogram, chroma features, Cepstral coefficient 
import time
X_all = []
Y_all = []
path="fold"
start = time.time()
for i in range(len(data)):
    fold_no=str(data.iloc[i]["fold"])
    file=data.iloc[i]["slice_file_name"]
    label=data.iloc[i]["classID"]
    filename=path+fold_no+"/"+file
    y,sr=librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
    X_all.append(features)
    Y_all.append(label)
end = time.time()

print(end-start)



## saving extracted features and labels
### extracting features and labels will take almost 1 hour, so I choose to save them all
XX =[]
for i in range(len(X_all)):
    XX.append(np.ravel(X_all[i]))

#saving the data numpy arrays
np.savetxt("X_all.csv", XX, delimiter=",")
np.savetxt("Y_all.csv",Y_all,delimiter=",")









##################################################################################################
##################################### model building #############################################
##################################################################################################


##building CNN

def CNN_model(X_train, Y_train, X_test, Y_test, k):
    X_train = X_train.reshape(len(X_train), 20,10, 1)
    X_test = X_test.reshape(len(X_test), 20,10, 1)
    input_dim = (20,10, 1)
    dropout_rate = 0.5
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_dim))
    # model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPool2D())
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    # model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPool2D())
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, Y_train, epochs = 35, batch_size = 100, validation_data = (X_test, Y_test))

    print(model.summary())
    predictions = model.predict(X_test)
    score = model.evaluate(X_test, Y_test)
    print(score)

    preds = np.argmax(predictions, axis = 1)
    result = pd.DataFrame(preds)
    result.to_csv("UrbanSound8kResults" + str(k) + ".csv")
    return score[1]


## normoalization function 
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data


## cross validation on CNN
def CV_CNN():
    X = pd.read_csv('X_all.csv', header = None)
    X = np.array(audio_norm(X))
    Y = pd.read_csv('Y_all.csv', header = None)
    Y = np.array(Y)

    folder = pd.read_csv("UrbanSound8K.csv")["fold"]
    folder = np.array(folder)
    acc = 0
    ## cross validation loop 
    for k in range(1, 11):  # 10-fold
        idx = np.argwhere(folder!=k).reshape(-1)
        X_train = X[idx]
        Y_train = Y[idx]
        idx = np.argwhere(folder==k).reshape(-1)
        X_test = X[idx]
        Y_test = Y[idx]

        acc += CNN_model(X_train, Y_train, X_test, Y_test, k)
    acc /= k
    print("Avg Accuracy:", acc)






## building SVM model 

 def model_SVM():
    start = time.time()
    C_range = [0.001,0.01,0.1,1,10,100,1000]
    Accuracy_list=[]
    Avg_score = []

    X = pd.read_csv('X_all.csv', header = None)
    X = np.array(audio_norm(X))
    Y = pd.read_csv('Y_all.csv', header = None)
    Y = np.array(Y)

    folder = pd.read_csv("UrbanSound8K.csv")["fold"]
    folder = np.array(folder)

    ## cross validation loop 
    for k in range(1, 11):  # 10-fold
        idx = np.argwhere(folder!=k).reshape(-1)
        X_train = X[idx]
        Y_train = Y[idx]
        idx = np.argwhere(folder==k).reshape(-1)
        X_test = X[idx]
        Y_test = Y[idx]

        ## grid search loop
        for c in C_range:
            RBF_SVM = SVC(C=c)
            RBF_SVM.fit(X_train,Y_train)
            pred = RBF_SVM.predict(X_test)
            Accuracy_list.append(accuracy_score(Y_test, pred))
        Max_accuracy=max(Accuracy_list)
        Avg_score.append(Max_accuracy)
    avg_accuarcy = np.mean(Avg_score)
    end = time.time()
    print(avg_accuarcy, '--------------', end-start)









## building random forest model 

def model_RF():
    start = time.time()
    n_estimator = [50,100,200]
    Accuracy_list=[]
    Avg_score = []

    X = pd.read_csv('X_all.csv', header = None)
    X = np.array(audio_norm(X))
    Y = pd.read_csv('Y_all.csv', header = None)
    Y = np.array(Y)

    folder = pd.read_csv("UrbanSound8K.csv")["fold"]
    folder = np.array(folder)

    ## cross validation loop 
    for k in range(1, 11):  # 10-fold
        idx = np.argwhere(folder!=k).reshape(-1)
        X_train = X[idx]
        Y_train = Y[idx]
        idx = np.argwhere(folder==k).reshape(-1)
        X_test = X[idx]
        Y_test = Y[idx]
        
        ## grid search loop
        for n in n_estimator:
            print(n,'222222222222222222')

            rf = RandomForestClassifier(n_estimators = n)
            rf.fit(X_train,Y_train)
            pred = rf.predict(X_test)
            Accuracy_list.append(accuracy_score(Y_test, pred))
        Max_accuracy=max(Accuracy_list)
        Avg_score.append(Max_accuracy)
    avg_accuarcy = np.mean(Avg_score)
    end = time.time()
    print(avg_accuarcy, '--------------', end-start)







## building KNN model 

def model_KNN():
    start = time.time()
    K = [i for i in range(5,40,2)]
    Accuracy_list=[]
    Avg_score = []

    X = pd.read_csv('X_all.csv', header = None)
    X = np.array(audio_norm(X))
    Y = pd.read_csv('Y_all.csv', header = None)
    Y = np.array(Y)

    folder = pd.read_csv("UrbanSound8K.csv")["fold"]
    folder = np.array(folder)

    ## cross validation loop 
    for n in range(1, 11):  # 10-fold
        idx = np.argwhere(folder!=n).reshape(-1)
        X_train = X[idx]
        Y_train = Y[idx]
        idx = np.argwhere(folder==n).reshape(-1)
        X_test = X[idx]
        Y_test = Y[idx]

        ## grid search loop
        for k in K:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train,Y_train)
            pred = knn.predict(X_test)
            Accuracy_list.append(accuracy_score(Y_test, pred))
        Max_accuracy=max(Accuracy_list)
        Avg_score.append(Max_accuracy)
    avg_accuarcy = np.mean(Avg_score)
    end = time.time()
    print(avg_accuarcy, '--------------', end-start)











## building ensemble model

def model_ensemble():
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier(n_neighbors=7)
    clf5 = DecisionTreeClassifier(max_depth=40)

    estimators = [('lr', clf1), ('rf', clf2), ('gnb', clf3), ('knn', clf4),('dt', clf5)]
    ensemble = VotingClassifier(estimators = estimators, voting='soft', weights=[2,100,2,1,1])
    
    X = pd.read_csv('X_all.csv', header = None)
    X = np.array(audio_norm(X))
    Y = pd.read_csv('Y_all.csv', header = None)
    Y = np.array(Y)
    Avg_score = []
    folder = pd.read_csv("UrbanSound8K.csv")["fold"]
    folder = np.array(folder)

    ## cross validation loop 
    for k in range(1, 11):  # 10-fold
        idx = np.argwhere(folder!=k).reshape(-1)
        X_train = X[idx]
        Y_train = Y[idx]
        idx = np.argwhere(folder==k).reshape(-1)
        X_test = X[idx]
        Y_test = Y[idx]
        
        #fit model to training data
        ensemble.fit(X_train, Y_train)
        #test our model on the test data
        score  = ensemble.score(X_test, Y_test)
        Avg_score.append(score)

    avg_accuarcy = np.mean(Avg_score)
    end = time.time()
    print(avg_accuarcy, '--------------', end-start)
