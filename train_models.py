import pickle
import numpy as np
import pandas as pd

#==============================================================================
# DATA PREPROCESSING
#==============================================================================
from sklearn import metrics

def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()
    
    return file

def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()

data = load_pickle('opinion_fact_sentences.pickle')
data = pd.DataFrame(data)
data.head()

# Divide the dataset into train and test sets
data['is_train'] = np.random.uniform(0, 1, len(data)) <= 0.85
train, test = data[data['is_train']==True], data[data['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Exctract the features and y_labels we'll be using for training
columns_to_drop = ['sentence', 'is_train', 'y_label']
features = data.drop(columns_to_drop, axis=1).columns
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
rf_classifier.fit(train[features], y_train)

# Predict the test data
rf_preds = rf_classifier.predict(test[features])

# Create confusion matrix
pd.crosstab(test['y_label'], rf_preds, rownames=['Actual Ranks'], colnames=['Predicted Ranks'])
print('Accuracy with random forest:')
print(metrics.accuracy_score(test['y_label'], rf_preds))
# ~0.90 - 0.93 accuracy
    
# View a list of the features and their importance scores
feature_importance = list(zip(train[features], rf_classifier.feature_importances_))
feature_importance = sorted(feature_importance, key=operator.itemgetter(1), reverse=True)

print(feature_importance[:5])
# NNP  => Proper Noun, Singular (Names, Countries, Cities etc.)
# PRP  => Pronoun, Personal (he, she, I, me etc.)
# NORP => Entity, Nationalities or religious or political groups.
# NNS  => Noun, Plural (cats, houses, potatoes etc.)
# VBD  => Verb, Past Tense (was, been, wrote etc.)

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
svm_classifier.fit(train[features], y_train)

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

feature_scaling = 0
if feature_scaling:
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    train_scaled[features] = scaler.fit_transform(train_scaled[features])
    test_scaled = test.copy()
    test_scaled[features] = scaler.transform(test_scaled[features])

    lr_classifier = LogisticRegression(random_state=0, verbose=1, C=0.01)
    lr_classifier.fit(train_scaled[features], y_train)
    
    lr_pred = lr_classifier.predict(test_scaled[features])
else:
    lr_classifier = LogisticRegression(random_state=0, verbose=1, C=0.01)
    lr_classifier.fit(train[features], y_train)
    
    lr_pred = lr_classifier.predict(test[features])


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
    nn_classifier.fit(train_scaled[features], y_train)
    nn_preds = nn_classifier.predict(test_scaled[features])
else:
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 5), random_state=1)
    nn_classifier.fit(train[features], y_train)
    nn_preds = nn_classifier.predict(test[features])
    
print('Accuracy Score with svm:')
print(metrics.accuracy_score(test['y_label'], nn_preds))
# 0.921 Accuracy or around 0.920 with feature scaling

cm = metrics.confusion_matrix(test['y_label'], nn_preds)
print(cm)

#==============================================================================
# SAVE MODELS
#==============================================================================
save_pickle(rf_classifier, 'models/rf_classifier.pickle')
save_pickle(svm_classifier, 'models/svm_classifier.pickle')
save_pickle(lr_classifier, 'models/lr_classifier.pickle')
save_pickle(nn_classifier, 'models/nn_classifier.pickle')