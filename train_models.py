import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import optimize_features
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from time import time

def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()
    
    return file

def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()
    
def test_classifier(X_train, y_train, X_test, y_test, classifier):
    
    print('#' * 50)
    classifier_name = str(type(classifier).__name__)
    print("Testing " + classifier_name)
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    print("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    print("Predicting time {0}s".format(time() - now))

    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    print("=================== Results ===================")
    print("            Fact     Opinion                   ")
    print("F1       " + str(f1))
    print("Precision" + str(precision))
    print("Recall   " + str(recall))
    print("Accuracy " + str(accuracy))
    print("===============================================")

    return model, precision, recall, accuracy, f1

def cv(classifier, X_train, y_train):
    classifier_name = str(type(classifier).__name__)
    now = time()
    print("Crossvalidating " + classifier_name + "...")
    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
    print("Crosvalidation completed in {0}s".format(time() - now))
    print("Accuracy: " + str(accuracy[0]))
    print("Average accuracy: " + str(np.array(accuracy[0]).mean()))
    
    return accuracy
    
#==============================================================================
# DATA PREPROCESSING
#==============================================================================
data = load_pickle('opinion_fact_sentences.pickle')
data = pd.DataFrame(data)
data.head()

# Divide the dataset into train and test sets
data['is_train'] = np.random.uniform(0, 1, len(data)) <= 0.85
train, test = data[data['is_train']==True], data[data['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Feature selection and optimization
# 0 - use all features
# 1 - only dependency tags
# 2 - only entity tags
# 3 - only part of speech tags
# 4 - top features from every label, as returned by random forest
# 5 - keep only entities and POS, initial idea
pick_features = 5
features = optimize_features.test_features(data, pick_features)
    
y_train = pd.factorize(train['y_label'])[0]

#==============================================================================
# # Train the RANDOM FOREST classifier
#==============================================================================
from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter
import operator

# Train the classifier
rf_classifier = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1,random_state=50, 
                             max_features=None)
rf_classifier, precision, recall, accuracy, f1 = test_classifier(train[features], y_train, test[features], test['y_label'], rf_classifier)
rf_acc = cv(rf_classifier, train[features], y_train)

# Learing time 33.64654469490051s
# Predicting time 0.3398551940917969s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.88263555  0.92587776]
# Precision[ 0.91205674  0.90739167]
# Recall   [ 0.85505319  0.94513274]
# Accuracy 0.909139213603
# 
# Crosvalidation completed in 127.3513731956482s
# CV Accuracy: [ 0.89548872  0.90827068  0.90827068  0.9112782   0.90451128  0.9075188
#                0.9075188   0.88713318]
# Average accuracy: 0.903748790713

# View a list of the features and their importance scores
feature_importance = list(zip(train[features], rf_classifier.feature_importances_))
feature_importance = sorted(feature_importance, key=operator.itemgetter(1), reverse=True)

print(feature_importance[:10])

prediction, bias, contributions = treeinterpreter.predict(rf_classifier, test.iloc[1:2][features])
print("Prediction", prediction)
print("Bias (trainset prior)", bias)
print("Feature contributions:")
for contrib, feature in zip(contributions[0], test[features].columns):
    print(feature, contrib)
    
#==============================================================================
# SUPPORT VECTOR MACHINE
#==============================================================================
from sklearn import svm

svm_classifier = svm.SVC()
svm_classifier, precision, recall, accuracy, f1 = test_classifier(train[features], y_train, test[features], test['y_label'], svm_classifier)
svm_acc = cv(svm_classifier, train[features], y_train)

# Learing time 5.493837594985962s
# Predicting time 0.6094698905944824s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.89821183  0.93576389]
# Precision[ 0.93956835  0.91047297]
# Recall   [ 0.86034256  0.9625    ]
# Accuracy 0.921234699308
#
# Crosvalidation completed in 18.2373263835907s
# CV Accuracy: [ 0.91278195  0.9112782   0.93007519  0.92180451  0.92105263  0.90977444
#                0.92030075  0.90293454]
# Average accuracy: 0.916250275802

svm_preds = svm_classifier.predict(test[features])
print('Accuracy Score with svm:')
print(metrics.accuracy_score(test['y_label'], svm_preds))
# ~0.91 - 0.93 accuracy

cm = metrics.confusion_matrix(test['y_label'], svm_preds)
print(cm)

#==============================================================================
# LINEAR REGRESSION
#==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

feature_scaling = 1
if feature_scaling:
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    train_scaled[features] = scaler.fit_transform(train_scaled[features])
    test_scaled = test.copy()
    test_scaled[features] = scaler.transform(test_scaled[features])

    lr_classifier = LogisticRegression(random_state=0, verbose=1, C=0.01)
    
    lr_classifier_scaled, precision, recall, accuracy, f1 = test_classifier(train_scaled[features], y_train, 
                                                                     test_scaled[features], test['y_label'], 
                                                                     lr_classifier)
    lr_acc_scaled = cv(lr_classifier, train_scaled[features], y_train)
    
    lr_pred = lr_classifier.predict(test_scaled[features])
    
    # [LibLinear]Learing time 0.046885013580322266s
    # Predicting time 0.0s
    # =================== Results ===================
    #             Fact     Opinion                   
    # F1       [ 0.69543147  0.86024845]
    # Precision[ 0.97163121  0.76098901]
    # Recall   [ 0.54150198  0.98928571]
    # Accuracy 0.808408728047
    #
    # Crossvalidating LogisticRegression...
    # Crosvalidation completed in 4.9046220779418945s
    # CV Accuracy: [ 0.79849624  0.79774436  0.8112782   0.8         0.79548872  0.80902256
    #   0.80526316  0.80361174]
    # Average accuracy: 0.802613121404
else:
    lr_classifier = LogisticRegression(random_state=0, verbose=1, C=0.01)
    
    lr_classifier, precision, recall, accuracy, f1 = test_classifier(train[features], y_train, 
                                                                     test[features], test['y_label'], 
                                                                     lr_classifier)
    lr_acc = cv(lr_classifier, train[features], y_train)
    
    lr_pred = lr_classifier.predict(test[features])
    
    # [LibLinear]Learing time 0.07819032669067383s
    # Predicting time 0.0s
    # =================== Results ===================
    #             Fact     Opinion                   
    # F1       [ 0.86823856  0.917962  ]
    # Precision[ 0.91654466  0.88879599]
    # Recall   [ 0.82476943  0.94910714]
    # Accuracy 0.898882384247
    #
    # Crossvalidating LogisticRegression...
    # Crosvalidation completed in 4.9425435066223145s
    # CV Accuracy: [ 0.89398496  0.9         0.90601504  0.89774436  0.90075188  0.89548872
    #   0.90827068  0.89014296]
    # Average accuracy: 0.899049825467

print('Accuracy with logistic regression:')
print(metrics.accuracy_score(test['y_label'], lr_pred))
# Around 0.912 accuracy or 0.807 with feature scaling

cm = metrics.confusion_matrix(test['y_label'], lr_pred)
print(cm)

#==============================================================================
# NEURAL NET CLASSIFIER
#==============================================================================
from sklearn.neural_network import MLPClassifier

if feature_scaling:
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 5), random_state=1)
    nn_classifier_scaled, precision, recall, accuracy, f1 = test_classifier(train_scaled[features], y_train, 
                                                                     test_scaled[features], test['y_label'], 
                                                                     nn_classifier)
    nn_acc_scaled = cv(nn_classifier, train_scaled[features], y_train)
    nn_preds = nn_classifier.predict(test_scaled[features])
    
    # Learing time 44.22556495666504s
    # Predicting time 0.03125405311584473s
    # =================== Results ===================
    #             Fact     Opinion                   
    # F1       [ 0.90261921  0.9360952 ]
    # Precision[ 0.92054795  0.92428198]
    # Recall   [ 0.88537549  0.94821429]
    # Accuracy 0.922831293241
    #
    # Crossvalidating MLPClassifier...
    # Crosvalidation completed in 164.07786178588867s
    # CV Accuracy: [ 0.91428571  0.93233083  0.93007519  0.92706767  0.92631579  0.92330827
    #             0.91654135  0.91647856]
    # Average accuracy: 0.923300420917
    
else:
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 5), random_state=1)
    nn_classifier, precision, recall, accuracy, f1 = test_classifier(train[features], y_train, 
                                                                     test[features], test['y_label'], 
                                                                     nn_classifier)
    nn_acc = cv(nn_classifier, train[features], y_train)
    nn_preds = nn_classifier.predict(test[features])
    
    # Learing time 48.68243479728699s
    # Predicting time 0.01562952995300293s
    # =================== Results ===================
    #             Fact     Opinion                   
    # F1       [ 0.87608841  0.9183223 ]
    # Precision[ 0.89100817  0.90829694]
    # Recall   [ 0.86166008  0.92857143]
    # Accuracy 0.901543374135
    #
    # Crossvalidating MLPClassifier...
    # Crosvalidation completed in 185.08190512657166s
    # Accuracy: [ 0.91654135  0.9112782   0.92180451  0.91879699  0.9075188   0.91578947
    #   0.91428571  0.90293454]
    # Average accuracy: 0.913618696855
    
print('Accuracy Score with neural net:')
print(metrics.accuracy_score(test['y_label'], nn_preds))
# 0.921 Accuracy or around 0.920 with feature scaling (up to 0.93 with dependency features)

cm = metrics.confusion_matrix(test['y_label'], nn_preds)
print(cm)

#==============================================================================
# SAVE MODELS
#==============================================================================
save_pickle(rf_classifier, 'models/rf_classifier.pickle')
save_pickle(svm_classifier, 'models/svm_classifier.pickle')
save_pickle(lr_classifier, 'models/lr_classifier.pickle')
save_pickle(lr_classifier_scaled, 'models/lr_classifier_scaled.pickle')
save_pickle(nn_classifier_scaled, 'models/nn_classifier_scaled.pickle')
save_pickle(nn_classifier, 'models/nn_classifier.pickle')
save_pickle(scaler, 'models/scaler.pickle')