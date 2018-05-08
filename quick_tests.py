import pickle
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_lg')

def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()
    
    return file

def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()

def divide_into_sentences(document):
    return [sent for sent in document.sents]

def number_of_fine_grained_pos_tags(sent):
    """
    Find all the tags related to words in a given sentence. Slightly more
    informative then part of speech tags, but overall similar data.
    Only one might be necessary. 
    For complete explanation of each tag, visit: https://spacy.io/api/annotation
    """
    tag_dict = {
    '-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0, 
    '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
    'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0, 
    'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0, 
    'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 
    'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 
    'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
    'OOV': 0, 'TRAILING_SPACE': 0}
    
    for token in sent:
        if token.is_oov:
            tag_dict['OOV'] += 1
        elif token.tag_ == '':
            tag_dict['TRAILING_SPACE'] += 1
        else:
            tag_dict[token.tag_] += 1
            
    return tag_dict

def number_of_dependency_tags(sent):
    """
    Find a dependency tag for each token within a sentence and add their amount
    to a distionary, depending how many times that particular tag appears.
    """
    dep_dict = {
    'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
    'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
    'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
    'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
    'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
    'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
    'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
    'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
    'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0, 
    'TRAILING_DEP': 0}
    
    for token in sent:
        if token.dep_ == '':
            dep_dict['TRAILING_DEP'] += 1
        else:
            try:
                dep_dict[token.dep_] += 1
            except:
                print('Unknown dependency for token: "' + token.orth_ +'". Passing.')
        
    return dep_dict

def number_of_specific_entities(sent):
    """
    Finds all the entities in the sentence and returns the amont of 
    how many times each specific entity appear in the sentence.
    """
    entity_dict = {
    'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
    'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
    'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
    'ORDINAL': 0, 'CARDINAL': 0 }
    
    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        entity_dict[entity] += 1
        
    return entity_dict

def sample(test_sent, classifier, scaler=None):
    # Preprocess using spacy
    parsed_test = divide_into_sentences(nlp(test_sent))
    
    # Get features
    sentence_with_features = {}
    entities_dict = number_of_specific_entities(parsed_test[0])
    sentence_with_features.update(entities_dict)
    pos_dict = number_of_fine_grained_pos_tags(parsed_test[0])
    sentence_with_features.update(pos_dict)
    #dep_dict = number_of_dependency_tags(parsed_test[0])
    #sentence_with_features.update(dep_dict)
    
    df = pd.DataFrame(sentence_with_features, index=[0])
    
    if scaler:
        df = scaler.transform(df)
    
    prediction = classifier.predict(df)
        
    # Run a prediction
    if prediction == 0:
        print('Your sentence: "' + test_sent + '" is a FACT!')
    else:
        print('Your sentence: "' + test_sent + '" is an OPINION!')

#==============================================================================
# Load models and test on random sentences
#==============================================================================
rf_classifier = load_pickle('models/rf_classifier.pickle')
svm_classifier = load_pickle('models/svm_classifier.pickle')
lr_classifier = load_pickle('models/lr_classifier.pickle')
nn_classifier = load_pickle('models/nn_classifier.pickle')
nn_classifier_scaled = load_pickle('models/nn_classifier_scaled.pickle')
scaler = load_pickle('models/scaler.pickle')

# Bunch of tests with different classifiers and sentences
test_sent = 'As far as I am concerned, donuts are amazing.'
sample(test_sent, nn_classifier_scaled, scaler)

test_sent = 'Donuts are torus-shaped, deep fried desserts, very often with a jam feeling on the inside.'
sample(test_sent, nn_classifier_scaled, scaler)

test_sent = 'Doughnut can also be spelled as "Donut", which is an American variant of the word.'
sample(test_sent, nn_classifier_scaled, scaler)

test_sent = 'This new graphics card I bought recently is pretty amazing, it has no trouble rendering my 3D donuts art in high quality.'
sample(test_sent, nn_classifier_scaled, scaler)

test_sent = 'Noone knows what are the origins of donuts.'
sample(test_sent, nn_classifier_scaled, scaler)

test_sent = 'The earliest origins to the modern doughnuts are generally traced back to the olykoek ("oil(y) cake"), which Dutch settlers brought with them to early New York'
sample(test_sent, nn_classifier_scaled, scaler)

test_sent = 'This donut is quite possibly the best tasting donut in the entire world.'
sample(test_sent, nn_classifier_scaled, scaler)
