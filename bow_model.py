import pickle
import numpy as np
import pandas as pd
import re
import nltk
import os
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
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
        word_df = pd.read_csv('bow_models/wordlist.csv')
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
    print("===============================================")
    classifier_name = str(type(classifier).__name__)
    now = time()
    print("Crossvalidating " + classifier_name + "...")
    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
    print("Crosvalidation completed in {0}s".format(time() - now))
    print("Accuracy: " + str(accuracy[0]))
    print("Average accuracy: " + str(np.array(accuracy[0]).mean()))
    print("===============================================")
    
    return accuracy

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

#==============================================================================
# Classification Time
#==============================================================================
# Split the data
X_train, X_test, y_train, y_test = train_test_split(bow.iloc[:, 1:], bow.iloc[:, 0],
                                                    train_size=0.80, stratify=bow.iloc[:, 0],
                                                    random_state=seed)

# BOW + Naive Bayes
from sklearn.naive_bayes import BernoulliNB
nb_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
nb_acc = cv(BernoulliNB(), bow.iloc[:,1:], bow.iloc[:,0])
#==============================================================================
# Testing BernoulliNB
# Learing time 1.833298921585083s
# Predicting time 0.3890223503112793s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.97222222  0.98237598]
# Precision[ 0.99578504  0.96784566]
# Recall   [ 0.94974874  0.99734924]
# Accuracy 0.978434504792
# ===============================================
#==============================================================================

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            RandomForestClassifier(random_state=seed, 
                                                                                   n_estimators=403, 
                                                                                   n_jobs=-1))
rf_acc = cv(RandomForestClassifier(n_estimators=403, n_jobs=-1, random_state=seed), bow.iloc[:, 1:], bow.iloc[:, 0])
#==============================================================================
# Testing RandomForestClassifier
# Learing time 63.64042282104492s
# Predicting time 0.69962477684021s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.93894305  0.9610984 ]
# Precision[ 0.9591195  0.9483871]
# Recall   [ 0.91959799  0.97415507]
# Accuracy 0.952476038339
#==============================================================================

# Suppor Vector Machine
from sklearn import svm
svm_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            svm.SVC())
svm_acc = cv(svm.SVC(), bow.iloc[:, 1:], bow.iloc[:, 0])
#==============================================================================
# Testing SVC
# Learing time 569.7381541728973s
# Predicting time 115.31103110313416s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.          0.75205582]
# Precision[ 0.          0.60263578]
# Recall   [ 0.  1.]
# Accuracy 0.602635782748
#==============================================================================

# Logistic Regression (since we're using BOW (binary features), there is not need for feature scaling)
from sklearn.linear_model import LogisticRegression
lr_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            LogisticRegression(random_state=0, verbose=1, C=0.01))
lr_acc = cv(LogisticRegression(random_state=0, verbose=1, C=0.01), bow.iloc[:, 1:], bow.iloc[:, 0])

#==============================================================================
# Testing LogisticRegression
# [LibLinear]Learing time 0.7871778011322021s
# Predicting time 0.0686790943145752s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.8631699   0.92624462]
# Precision[ 0.99736495  0.86361032]
# Recall   [ 0.76080402  0.99867462]
# Accuracy 0.904153354633
#
# CV Accuracy: [ 0.90990415  0.91373802  0.91693291  0.90926518  0.90990415  0.91246006
#                0.91304348  0.90025575]
# Average accuracy: 0.910687963198
#==============================================================================

# Neural Network
from sklearn.neural_network import MLPClassifier
nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256, 3), random_state=1)
nn_model, precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, 
                                                            nn_classifier)
nn_acc = cv(nn_classifier, bow.iloc[:, 1:], bow.iloc[:, 0])

#==============================================================================
# Testing MLPClassifier
# Learing time 63.365925550460815s
# Predicting time 0.16393327713012695s
# =================== Results ===================
#             Fact     Opinion                   
# F1       [ 0.96911392  0.9798879 ]
# Precision[ 0.97653061  0.97506562]
# Recall   [ 0.96180905  0.98475812]
# Accuracy 0.975638977636
#
# Crosvalidation completed in 688.3820369243622s
# CV Accuracy: [ 0.96996805  0.96613419  0.97827476  0.96613419  0.97060703  0.96932907
#                0.9629156   0.97826087]
# Average accuracy: 0.970202969367
#==============================================================================

save_pickle(nn_model, 'bow_pickles/nn_model.pickle')
save_pickle(lr_model, 'bow_pickles/lr_model.pickle')
save_pickle(svm_model, 'bow_pickles/svm_model.pickle') 
save_pickle(rf_model, 'bow_pickles/rf_model.pickle')
save_pickle(nb_model, 'bow_pickles/nb_model.pickle')

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
    casual_test(test_sent, lr_model, wordlist)
