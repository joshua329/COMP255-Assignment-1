import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt 
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def load_dataset (i)
    df = pd.read_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/dataset_' + str(fn) + '.txt', sep=',', header=None)
    return df
    
def visualize_data(df): # Sprint 1
    for fn in range (1, 20):
        print ("---------------------------------------------------------") # indicates a new file
        print ("File: data-set " + str(fn))
        df = input_Data(fn)
        for i in range (1, 14): #upper range is not inclusive
            Recognition_Activity = df[df[24] == i].values # getting the human activity code
            print ("Active Code: " + str(i))
            Recognition_Activity = noise_removing(b, a, Recognition_Activity[::])
            for j in range (1, 7): # All the values of the first sensor
                plt.plot(noise_removing(b, a, Recognition_Activity[:, j - 1: j]))
                plt.show()

def remove_signal_noises(arr):
    b, a = signal.butter(3, 0.04, 'low', analog=False) # 3rd order, filtering with lowpass
    return signal.lfilter(b, a, arr);

def prepare_training_set (max_Range, training_Data):
    train_Data = np.empty(shape=(0, 10))
    for s in range(max_Range):
        if s < max_Range - 1:
            sample_data = training_data[1000*s:1000*(s + 1), :]
        else:
            sample_data = training_data[1000*s:, :]
            features = []
            for i in range(3):
                features.append(np.min(sample_data[:, i])) # gets min value
                features.append(np.max(sample_data[:, i])) # get max value 
                features.append(df.std(sample_data[:, i])) # gets standard deviation
            features.append(sample_data[0, -1])
            features = np.array([features])
            train_Data = np.concatenate((train_Data, features), axis=0)
    return train_Data
    
def extract_features ():
    train_Data = np.empty(shape=(0, 10))
    test_data = np.empty(shape=(0, 10))
    for i in range(1, 20):
        df = input_Data(i) # reading input files
        for c in range(1, 14):
            Recognition_Activity = df[df[24] == c].values
            for j in range(24):
                Recognition_Activity[:, j] = remove_signal_noises(Recognition_Activity[:, j])
            datat_len = len(Recognition_Activity)
            training_len = math.floor(datat_len * 0.8)
            training_data = Recognition_Activity[:training_len, :]
            testing_data = Recognition_Activity[training_len:, :]
            training_sample_number = training_len // 1000 + 1
            testing_sample_number = (datat_len - training_len) // 1000 + 1
            
            train_Data = prepare_training_set (training_sample_number, training_data);
            testing = prepare_training_set (training_sample_number, training_data);

    df_training = pd.DataFrame(training)
    df_testing = pd.DataFrame(testing)
    
    df_training.to_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/training_data.csv', index=None, header=None)
    df_testing.to_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/testing_data.csv', index=None, header=None)

def test_the_given_models(arr_values, subtract):
    return (df_testing[9].values) - int(subtract)
    
def training_the_given_models():
    df_training = pd.read_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/training_data.csv', header=None)
    df_testing = pd.read_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/testing_data.csv', header=None)
    # training variables
    y_train = test_the_given_models(df_training[9].values, 1) 
    X_train = (df_training.drop([9], axis=1)).values

    y_test = test_the_given_models(df_testing[9].values, 1)
    X_test = (df_testing.drop([9], axis=1)).values
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN Algorithm
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    # We could use confusion matrix to view the classification for each activity.
    print(confusion_matrix(y_test, y_pred))
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2, 1e-3, 1e-4],
                     'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 100]},
                    {'kernel': ['linear'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}]
    acc_scorer = make_scorer(accuracy_score)
    grid_obj  = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=acc_scorer)
    grid_obj  = grid_obj .fit(X_train, y_train)
    
    # SVM Algorit
    clf = grid_obj.best_estimator_
    print('best clf:', clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
