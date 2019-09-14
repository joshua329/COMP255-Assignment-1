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

#Commented by: James Xi Zheng 12/Aug/2019
#please create functions to do the following jobs
#1. load dataset ->  sample code availalable in the workshops
#2. visualize data -> sample code given
#3. remove signal noises -> sample code given
#4. extract features -> sample code given
#5. prepare training set -> sample code given 
#6. training the given models -> sample code given
#7. test the given models -> sample code given
#8. print out the evaluation results -> sample code given

#as I said in the lecture, the sample code is completed in a un-professional software engineering style
#software refactoring is required
#please manage the project using SCRUM sprints and manage the source code using Github
#document your progress and think critically what are missing from such IoT application and what are missing to move such IoT application from PoC (proof of concept) to solve real-world life
#think with which components added, what kind of real-world problems can be solved by it -> this shall be discussed in the conclusion part in the document

def load_dataset (i)
    df = pd.read_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/dataset_' + str(fn) + '.txt', sep=',', header=None)
    return df
    
def visualize_data(df): # Sprint 1
    for fn in range (1, 20):
        print ("---------------------------------------------------------") # indicates a new file
        print ("File: data-set " + str(fn))
        df = input_Data(fn)
    b, a = signal.butter(3, 0.04, 'low', analog=False)
    
    for i in range (1, 14): #upper range is not inclusive
        Recognition_Activity = df[df[24] == i].values # getting the human activity code
        print ("Active Code: " + str(i))
        Recognition_Activity = noise_removing(b, a, Recognition_Activity[::])
        for j in range (1, 7):
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
        df = input_Data(i)
        for c in range(1, 14):
            Recognition_Activity = df[df[24] == c].values
            b, a = signal.butter(3, 0.04, 'low', analog=False)
            for j in range(24):
                Recognition_Activity[:, j] = signal.lfilter(b, a, activity_data[:, j])
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
    
def training_the_given_models():
    df_training = pd.read_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/training_data.csv', header=None)
    df_testing = pd.read_csv('C:/Users/CHARLIE/Desktop/COMP255/dataset/testing_data.csv', header=None)

    y_train = df_training[9].values # Labels should start from 0 in sklearn
    y_train = y_train - 1
    df_training = df_training.drop([9], axis=1)
    X_train = df_training.values

    y_test = df_testing[9].values
    y_test = y_test - 1
    df_testing = df_testing.drop([9], axis=1)
    X_test = df_testing.values
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Build KNN classifier, in this example code
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
    clf = grid_obj.best_estimator_
    print('best clf:', clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
