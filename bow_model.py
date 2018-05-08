import pickle
import numpy as np
import pandas as pd
import re
import nltk
import os
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from time import time

import random
seed = 2
random.seed(seed)

def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()
    
    return file

def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()
    
def preprocess_data(data, stemmer=nltk.PorterStemmer()):
    special_chars = [",", ":", "\"", "=", "&", ";", "%", "$", "@", "%", "^", "*", "(", ")", "{", "}",
                 "[", "]", "|", "/", "\\", ">", "<", "-",  "!", "?", ".", "'", "--", "---", "#"]
    
    for example in data:
        # Remove numbers
        example['sentence'] = re.sub(r'\s?[0-9]+\.?[0-9]*', '', example['sentence'])
        
        # Remove special characters
        for remove in special_chars:
            example['sentence'] = example['sentence'].replace(remove, '')
            
        # Tokenize and stem the data
        example['tokenized_sentence'] = nltk.word_tokenize(example['sentence'])
        example['stemmed_sentence'] = [stemmer.stem(token.lower()) for token in example['tokenized_sentence']]
        
    return data

def build_wordlist(data, min_occurrences=3, max_occurrences=500, stopwords=nltk.corpus.stopwords.words('english'), whitelist=None):
    wordlist = []
    
    # If the wordlist file has been created earlier, use this one
    if os.path.isfile('bow_models/wordlist.csv'):
        with open('bow_models/wordlist.csv', 'r') as f:
            word_df = pd.read_csv(f)
            word_df = word_df[word_df['occurrences'] > min_occurrences]
            wordlist = list(word_df.loc[:, 'word'])
            return
    
    # Else create a new one
    words = Counter()
    for index in data.index:
        words.update(data.loc[index, 'stemmed_sentence'])
        
    for index, stop_word in enumerate(stopwords):
        if stop_word not in whitelist:
            del words[stop_word]
            
    word_df = pd.DataFrame(data={'word': [key for key, value in words.most_common() if min_occurrences < value < max_occurrences],
                                 'occurrences': [value for _, value in words.most_common() if min_occurrences < value < max_occurrences]},
                           columns=['word', 'occurrences'])
    word_df.to_csv('bow_models/wordlist.csv', index_label='index')
     
    wordlist = [key for key, value in words.most_common() if min_occurrences < value < max_occurrences]
    return wordlist

def build_bow_model(data, wordlist, testset=False):
    label_column = []
    if not testset:
        label_column = ['label']
        
    columns = label_column + list(map(lambda word: word+'_bow', wordlist))
    labels = []
    rows = []
    
    for index in data.index:
        current_row = []
        
        if not testset:
            # Add label
            current_label = data.loc[index, 'y_label']
            labels.append(current_label)
            current_row.append(current_label)
            
        # Add BOW
        tokens = set(data.loc[index, 'stemmed_sentence'])
        for word in wordlist:
            current_row.append(1 if word in tokens else 0)
            
        rows.append(current_row)
        
    data_model = pd.DataFrame(rows, columns=columns)
    data_labels = pd.Series(labels)
    
    if not testset:
        assert len(data_model) == len(data_labels)
        return data_model, data_labels
    else:
        return data_model
    
def casual_test(sentence, classifier, wordlist):
    print('#' * 50)
    print('Sentence: ' + sentence)
    test_dict = {}
    test_list = []
    test_dict['sentence'] = sentence
    test_list.append(test_dict)
    
    test_list = preprocess_data(test_list)
    test_list = pd.DataFrame(test_list)
    bow = build_bow_model(test_list, wordlist, testset=True)
    
    pred = classifier.predict(bow)
    if pred[0] == 0:
        print('The above sentence is an OPINION!')
    else:
        print('The above sentence is a FACT!')

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
# Load up and preprocess the data    
#==============================================================================
# Create data with no sentence structure features anymore
data = load_pickle('opinion_fact_sentences.pickle')
new_data = []
for example in data:
    new_example = {}
    new_example['sentence'] = example['sentence']
    new_example['y_label'] = example['y_label']
    new_example['is_train'] = example['is_train']
    new_data.append(new_example)
assert len(new_data) == len(data)

# For BoW model, we'll remove any special characters and numbers as well as tokenize the sentences and stem them
data = preprocess_data(new_data)
data = pd.DataFrame(data)
data.head()

# Build the wordlist dictionary
words = Counter()
for index in data.index:
    words.update(data.loc[index, 'stemmed_sentence'])
words.most_common(10)

# Remove the stopwords
stopwords = nltk.corpus.stopwords.words('english')
whitelist = ['n\'t', 'not']

# Build a wordlist (or import one if found) for the data
wordlist = build_wordlist(data, whitelist=whitelist)
wordlist[:10]
len(wordlist)

#==============================================================================
# Transform the data into BOW model
#==============================================================================
bow, labels = build_bow_model(data, wordlist)
assert len(data) == len(bow) == len(labels)
bow.head()
labels.head()
data.head()

save_pickle(bow, 'bow_pickles/bow.pickle')
save_pickle(labels, 'bow_pickles/labels.pickle')
save_pickle(data, 'bow_pickles/data.pickle')

bow = load_pickle('bow_pickles/bow.pickle')
labels = load_pickle('bow_pickles/labels.pickle')
data = load_pickle('bow_pickles/data.pickle')
bow.iloc[0, 0]
#==============================================================================
# Classification Time
#==============================================================================
# Split the data
X_train, X_test, y_train, y_test = train_test_split(bow.iloc[:, 1:], bow.iloc[:, 0],
                                                    train_size=0.85, stratify=bow.iloc[:, 0],
                                                    random_state=seed)
X_train = [] 
X_test = []
y_train = []
y_test = []
for index, example in enumerate(data['is_train']):
    if example == True:
        X_train.append(bow.iloc[index, 1:])
        y_train.append(bow.iloc[index, 0])
    else:
        X_test.append(bow.iloc[index, 1:])
        y_test.append(bow.iloc[index, 0])
assert len(X_test) == len(y_test)
assert len(X_train) == len(y_train)

save_pickle(X_train, 'bow_pickles/X_train.pickle')
save_pickle(X_test, 'bow_pickles/X_test.pickle')
save_pickle(y_train, 'bow_pickles/y_train.pickle')
save_pickle(y_test, 'bow_pickles/y_test.pickle')

#==============================================================================
# # BOW + Naive Bayes
#==============================================================================
from sklearn.naive_bayes import BernoulliNB
nb_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
nb_acc = cv(BernoulliNB(), bow.iloc[:, 1:], bow.iloc[:, 0])

nb_preds_bow = nb_model.predict(X_test)
nb_cm = metrics.confusion_matrix(y_test, nb_preds_bow)
print_cm(nb_cm)

#NEW WITH PROPER DIVISION ?
#               Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1080                   41           1121
# Actual Optinion:     5                   1105           1110
#                     1085                 1146


#==============================================================================
# # Random Forest
#==============================================================================
from sklearn.ensemble import RandomForestClassifier
rf_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            RandomForestClassifier(random_state=seed, 
                                                                                   n_estimators=403, 
                                                                                   n_jobs=-1))
rf_acc = cv(RandomForestClassifier(n_estimators=403, n_jobs=-1, random_state=seed), bow.iloc[:, 1:], bow.iloc[:, 0])

rf_preds_bow = rf_model.predict(X_test)
rf_cm = metrics.confusion_matrix(y_test, rf_preds_bow)
print_cm(rf_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1042                    79             1121
# Actual Optinion:     16                   1094           1110
#                     1058                   1173

#==============================================================================
# # Suppor Vector Machine
#==============================================================================
from sklearn import svm
svm_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            svm.SVC())
svm_acc = cv(svm.SVC(), bow.iloc[:, 1:], bow.iloc[:, 0])
save_pickle(svm_acc, 'bow_pickles/svm_acc.pickle')

svm_preds_bow = svm_model.predict(X_test)
svm_cm = metrics.confusion_matrix(y_test, svm_preds_bow)
print_cm(svm_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        10                    1111             1121
# Actual Optinion:     0                   1110           1110
#                     10                   2221

#==============================================================================
# # Logistic Regression (since we're using BOW (binary features), there is not need for feature scaling)
#==============================================================================
from sklearn.linear_model import LogisticRegression
lr_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            LogisticRegression(random_state=0, verbose=1, C=0.01))
lr_acc = cv(LogisticRegression(random_state=0, verbose=1, C=0.01), bow.iloc[:, 1:], bow.iloc[:, 0])

lr_preds_bow = lr_model.predict(X_test)
lr_cm = metrics.confusion_matrix(y_test, lr_preds_bow)
print_cm(lr_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1041                    80             1121
# Actual Optinion:     18                   1092           1110
#                     1059                   1172


#==============================================================================
# # Neural Network
#==============================================================================
from sklearn.neural_network import MLPClassifier
nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256, 3), random_state=1)
nn_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            nn_classifier)
nn_acc = cv(nn_classifier, bow.iloc[:, 1:], bow.iloc[:, 0])

nn_preds_bow = nn_model.predict(X_test)
nn_cm = metrics.confusion_matrix(y_test, nn_preds_bow)
print_cm(nn_cm)

#                Predicted Fact     Predicted Opinion      Total 
# Actual Fact:        1086                    35           1121
# Actual Optinion:     17                   1093           1110
#                     1103                   1128


save_pickle(nn_model, 'bow_pickles/nn_model.pickle')
save_pickle(lr_model, 'bow_pickles/lr_model.pickle')
save_pickle(svm_model, 'bow_pickles/svm_model.pickle') 
save_pickle(rf_model, 'bow_pickles/rf_model.pickle')
save_pickle(nb_model, 'bow_pickles/nb_model.pickle')

save_pickle(nb_preds_bow, 'bow_pickles/nb_preds_bow.pickle')
save_pickle(rf_preds_bow, 'bow_pickles/rf_preds_bow.pickle')
save_pickle(lr_preds_bow, 'bow_pickles/lr_preds_bow.pickle')
save_pickle(nn_preds_bow, 'bow_pickles/nn_preds_bow.pickle')
save_pickle(svm_preds_bow, 'bow_pickles/svm_preds_bow.pickle')

nb_model = load_pickle('bow_pickles/nb_model.pickle')
#==============================================================================
# Sanity Check on new data
#==============================================================================
test_sentences = []
test_sentences.append('As far as I am concerned, donuts are amazing.')
test_sentences.append('Donuts are torus-shaped, deep fried desserts, very often with a jam feeling on the inside.')
test_sentences.append('Doughnut can also be spelled as "Donut", which is an American variant of the word.')
test_sentences.append('This new graphics card I bought recently is pretty amazing, it has no trouble rendering my 3D donuts art in high quality.')
test_sentences.append('Noone knows what are the origins of donuts.')
test_sentences.append('The earliest origins to the modern doughnuts are generally traced back to the olykoek ("oil(y) cake"), which Dutch settlers brought with them to early New York')
test_sentences.append('This donut is quite possibly the best tasting donut in the entire world.')

for test_sent in test_sentences:
    casual_test(test_sent, nn_model, wordlist)
    
    
nn_model = load_pickle('bow_pickles/nn_model.pickle')
lr_model = load_pickle('bow_pickles/lr_model.pickle')
svm_model = load_pickle('bow_pickles/svm_model.pickle') 
rf_model = load_pickle('bow_pickles/rf_model.pickle')
nb_model = load_pickle('bow_pickles/nb_model.pickle')

nb_preds_bow = load_pickle('bow_pickles/nb_preds_bow.pickle')
rf_preds_bow = load_pickle('bow_pickles/rf_preds_bow.pickle')
lr_preds_bow = load_pickle('bow_pickles/lr_preds_bow.pickle')
nn_preds_bow = load_pickle('bow_pickles/nn_preds_bow.pickle')
svm_preds_bow = load_pickle('bow_pickles/svm_preds_bow.pickle')
