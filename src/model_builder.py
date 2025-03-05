from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import os

# Add the 'src' folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Importing DataPreprocessing from data_preprocess.py
from src.data_preprocess import DataPreprocessing


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)
        self.accuracy = None  # Store model accuracy

    def dt(self, X_train, X_test, y_train, y_test):
        """Train a Decision Tree model"""
        DT_classifier = DecisionTreeClassifier()
        DT_classifier.fit(X_train, y_train)
        DT_predicted = DT_classifier.predict(X_test)
        
        self.accuracy = accuracy_score(y_test, DT_predicted)
        return DT_classifier

    def train_mlp(self, X_train, X_test, y_train, y_test, hidden_layers, lr, max_iters):
        """Train an MLP Neural Network with given hyperparameters"""
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                            learning_rate_init=lr, 
                            max_iter=max_iters, 
                            random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        return self.accuracy  # ✅ FIXED: Return accuracy instead of the model
