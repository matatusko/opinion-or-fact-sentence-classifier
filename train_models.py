import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import optimize_features
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate 
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
    scoring = ['accuracy', 'f1', 'f1_micro', 'f1_macro', 'precision', 'recall']
    classifier_name = str(type(classifier).__name__)
    now = time()
    print("Crossvalidating " + classifier_name + "...")
    cv_results = [cross_validate(classifier, X_train, y_train, cv=8, n_jobs=-1, scoring=scoring, return_train_score=False)]
    print("Crosvalidation completed in {0}s".format(time() - now))
    #print("Accuracy: " + str(accuracy[0]))
    #print("Average accuracy: " + str(np.array(accuracy[0]).mean()))
    
    return cv_results

def print_cm(cm):
    print("               Predicted Fact     Predicted Opinion      Total ")
    print("Actual Fact:       ", cm[0][0], "                  ", cm[0][1], "           ", (cm[0][0] + cm[0][1]))
    print("Actual Optinion:    ", cm[1][0], "                 ",cm[1][1], "         ", (cm[1][0] + cm[1][1]))
    print("                   ", (cm[0][0] + cm[1][0]), "                 ", (cm[0][1] + cm[1][1]))
    
#==============================================================================
# DATA PREPROCESSING
#==============================================================================
is_train = np.random.uniform(0, 1, len(data)) <= 0.85
for index, example in enumerate(data):
    data[index]['is_train'] = is_train[index]
save_pickle(data, 'opinion_fact_sentences.pickle')

data = load_pickle('opinion_fact_sentences.pickle')
data = pd.DataFrame(data)
data.head()

count_opinion = 0
count_fact = 0
for example in data['y_label']:
    if example == 0:
        count_fact += 1
    else:
        count_opinion += 1
print('Got {} opinions and {} factual sentences.'.format(count_opinion, count_fact))
print('Total sentences: {}'.format(count_opinion + count_fact))

# Divide the dataset into train and test sets
train, test = data[data['is_train']==True], data[data['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))

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
assert(len(y_train) + len(test) == len(data))

real_values = test['y_label']
save_pickle(real_values, './real_test_values.pickle')
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
rf_cv_scores = cv(rf_classifier, data[features], data['y_label'])

feature_importance = list(zip(train[features], rf_classifier.feature_importances_))
feature_importance = sorted(feature_importance, key=operator.itemgetter(1), reverse=True)

print(feature_importance[:10])

prediction, bias, contributions = treeinterpreter.predict(rf_classifier, test.iloc[1:2][features])
print("Prediction", prediction)
print("Bias (trainset prior)", bias)
print("Feature contributions:")
for contrib, feature in zip(contributions[0], test[features].columns):
    print(feature, contrib)

rf_classifier = rf_classifier.fit(train[features], train['y_label'])
rf_preds_epos = rf_classifier.predict(test[features])
rf_cm = metrics.confusion_matrix(test['y_label'], rf_preds_epos)
print_cm(rf_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1007                    114             1121
# Actual Optinion:     102                   1008           1110
#                     1109                   1122

    
#==============================================================================
# NAIVE BAYES
#==============================================================================
from sklearn.naive_bayes import BernoulliNB
nb_classifier, precision, recall, accuracy, f1 = test_classifier(train[features], y_train, test[features], test['y_label'], BernoulliNB())
nb_cv_scores = cv(BernoulliNB(), data[features], data['y_label'])

nb_classifier = BernoulliNB().fit(train[features], train['y_label'])
nb_preds_epos = nb_classifier.predict(test[features])    
nb_cm = metrics.confusion_matrix(test['y_label'], nb_preds_epos)
print_cm(nb_cm )

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        895                    226             1121
# Actual Optinion:     77                   1033           1110
#                     972                   1259

#==============================================================================
# SUPPORT VECTOR MACHINE
#==============================================================================
from sklearn import svm

svm_classifier = svm.SVC()
svm_classifier, precision, recall, accuracy, f1 = test_classifier(train[features], y_train, test[features], test['y_label'], svm_classifier)
svm_acc = cv(svm_classifier, data[features], data['y_label'])

svm_preds = svm_classifier.predict(test[features])
print('Accuracy Score with svm:')
print(metrics.accuracy_score(test['y_label'], svm_preds))
# ~0.91 - 0.93 accuracy

svm_classifier = svm_classifier.fit(train[features], train['y_label'])
svm_preds_epos = svm_classifier.predict(test[features])    
svm_cm = metrics.confusion_matrix(test['y_label'], svm_preds_epos)
print_cm(svm_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1000                  121            1121
# Actual Optinion:     65                   1045           1110
#                     1065                  1166

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
    
    data_scaled = data.copy()
    data_scaled[features] = scaler.fit_transform(data[features])

    lr_classifier = LogisticRegression(random_state=0, verbose=1, C=0.01)
    
    lr_classifier_scaled, precision, recall, accuracy, f1 = test_classifier(train_scaled[features], y_train, 
                                                                     test_scaled[features], test['y_label'], 
                                                                     lr_classifier)
    lr_acc_scaled = cv(lr_classifier, data_scaled[features], data_scaled['y_label'])



print('Accuracy with logistic regression:')
print(metrics.accuracy_score(test['y_label'], lr_pred))
# Around 0.912 accuracy or 0.807 with feature scaling

lr_classifier_scaled = lr_classifier.fit(train_scaled[features], train_scaled['y_label'])
lr_preds_epos = lr_classifier_scaled.predict(test_scaled[features])    
lr_cm = metrics.confusion_matrix(test['y_label'], lr_preds_epos)
print_cm(lr_cm)

#                 Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        854                    267             1121
# Actual Optinion:     93                   1017           1110
#                     947                   1284

#==============================================================================
# NEURAL NET CLASSIFIER
#==============================================================================
from sklearn.neural_network import MLPClassifier

if feature_scaling:
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 5), random_state=1)
    nn_classifier_scaled, precision, recall, accuracy, f1 = test_classifier(train_scaled[features], y_train, 
                                                                     test_scaled[features], test['y_label'], 
                                                                     nn_classifier)
    nn_acc_scaled = cv(nn_classifier, data_scaled[features], data_scaled['y_label'])
    
    
print('Accuracy Score with neural net:')
print(metrics.accuracy_score(test['y_label'], nn_preds))
# 0.921 Accuracy or around 0.920 with feature scaling (up to 0.93 with dependency features)

nn_classifier_scaled = nn_classifier.fit(train_scaled[features], train_scaled['y_label'])
nn_preds_epos = nn_classifier_scaled.predict(test_scaled[features])    
nn_cm = metrics.confusion_matrix(test['y_label'], nn_preds_epos)
print_cm(nn_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1019                  102             1121
# Actual Optinion:     66                   1044           1110
#                     1085                  1146

#==============================================================================
# SAVE MODELS
#==============================================================================
save_pickle(rf_classifier, 'epos_pickles/rf_classifier.pickle')
save_pickle(svm_classifier, 'epos_pickles/svm_classifier.pickle')
#save_pickle(lr_classifier, 'models/lr_classifier.pickle')
save_pickle(lr_classifier_scaled, 'epos_pickles/lr_classifier_scaled.pickle')
save_pickle(nn_classifier_scaled, 'epos_pickles/nn_classifier_scaled.pickle')
#save_pickle(nn_classifier, 'models/nn_classifier.pickle')
save_pickle(scaler, 'epos_pickles/scaler.pickle')

save_pickle(nb_preds_epos, 'epos_pickles/nb_preds_epos.pickle')
save_pickle(rf_preds_epos, 'epos_pickles/rf_preds_epos.pickle')
save_pickle(lr_preds_epos, 'epos_pickles/lr_preds_epos.pickle')
save_pickle(nn_preds_epos, 'epos_pickles/nn_preds_epos.pickle')
save_pickle(svm_preds_epos, 'epos_pickles/svm_preds_epos.pickle')