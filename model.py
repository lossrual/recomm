import numpy as np
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.cross_validation
from sklearn.metrics.metrics import accuracy_score, classification_report

class event_recommend(object):
    def __init__(self):
        self.classifier = None
    
    def load_data(self, path):
        data = pd.read_csv(path)
        X = data[:, 0:-1]
        y = data[:, -1]
        return X, y
            
    def train_model(self, X, y):
        classifiers = [RandomForestClassifier(), SVC(kernel = 'rbf'),DecisionTreeClassifier()]
        self.classifier = classifiers[2]
        self.classifier.n_classes_ = 2
        self.classifier.fit(X, y)
        
    def cross_validation(self, X, y):
        clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        print cross_validation.cross_val_score(clf, X, y, cv=5)


    def predict_res(self, X):
        return self.classifier.predict(X)
        
    def run_model(self, train_path, test_path):
        trainx, trainy = self.load_data(train_path)
        self.train_model(trainx, trainy)
        testx, testy = self.load_data(test_path)
        predy = self.predict_res(testx)
        accuracy = accuracy_score(testy, predy) 
        label = [1, 0]
        classifier = ['interested', 'nointerested']
        result = classification_report(testy, predy, labels=label, target_names = classifier) + '\naccuracy\t' + str(accuracy)
        print result
        
if __name__ == '__main__':
    clf = event_recommend()
    clf.run_model(('feature.csv', 'test.csv')
    
