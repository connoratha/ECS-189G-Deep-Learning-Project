import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_test, predictions):
    metrics = {
        'Accuracy' : accuracy_score(y_test, predictions),
        'F1 Score (weighted)' : f1_score(y_test, predictions, average='weighted'),
        'Recall (weighted)' : recall_score(y_test, predictions, average='weighted'),
        'Precision (weighted)' : precision_score(y_test, predictions, average='weighted', zero_division=0),
        'F1 Score (macro)' : f1_score(y_test, predictions, average='macro'),
        'Recall (macro)' : recall_score(y_test, predictions, average='macro'),
        'Precision (macro)' : precision_score(y_test, predictions, average='macro', zero_division=0),
        'F1 Score (micro)' : f1_score(y_test, predictions, average='micro'),
        'Recall (micro)' : recall_score(y_test, predictions, average='micro'),
        'Precision (micro)' : precision_score(y_test, predictions, average='micro', zero_division=0)
    }

    return metrics

def print_metrics(metrics):
    print("Accuracy: ", metrics['Accuracy'], "\n")

    print("---Weighted Metrics---")
    print("F1 Score: ", metrics['F1 Score (weighted)'])
    print("Recall: ", metrics['Recall (weighted)'])
    print("Precision: ", metrics['Precision (weighted)'], "\n")

    print("---Macro Metrics---")
    print("F1 Score: ", metrics['F1 Score (macro)'])
    print("Recall: ", metrics['Recall (macro)'])
    print("Precision: ", metrics['Precision (macro)'], "\n")

    print("---Micro Metrics---")
    print("F1 Score: ", metrics['F1 Score (micro)'])
    print("Recall: ", metrics['Recall (micro)'])
    print("Precision: ", metrics['Precision (micro)'], "\n")

print("Method Running...")

# create dataframe 'train_data' from training dataset
train_file_path = 'C:\\Users\\conno\\Downloads\\UC Davis\\year 3\\Winter Quarter\\ECS 189G\\Project Stages\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_2_data\\train.csv'
train_data = pd.read_csv(train_file_path)

# create dataframe 'test_data' from testing dataset
test_file_path = 'C:\\Users\\conno\\Downloads\\UC Davis\\year 3\\Winter Quarter\\ECS 189G\\Project Stages\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_2_data\\test.csv'
test_data = pd.read_csv(test_file_path)

# initialize MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,50,), activation='relu', solver='sgd', max_iter=5000)    # learning rate is default 0.001

# define X_train and y_train
X_train = train_data.iloc[:, 1:].values   # all values
y_train = train_data.iloc[:, 0].values    # all labels

# define X_test and y_test
X_test = test_data.iloc[:, 1:].values   # all values
y_test = test_data.iloc[:, 0].values    # all labels

# find best parameters
'''
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')
print("---Training Data---")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)

print("---Testing Method---")
predictions = best_model.predict(X_train)

print("---Gathering Results---")
metrics = evaluate_model(y_test, predictions)
print_metrics(metrics)
'''

# train data
print("---Training Data---")
mlp.fit(X_train, y_train)

print("---Testing Method---")
predictions = mlp.predict(X_test)

print("---Gathering Results---")
metrics = evaluate_model(y_test, predictions)
print_metrics(metrics)


# Creating plot
print("---Generating Plot---")
plt.figure()
plt.title("MLP Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")

train_sizes, train_scores, test_scores = learning_curve(mlp, X_train, y_train, train_sizes=np.linspace(.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.grid()
plt.show()
