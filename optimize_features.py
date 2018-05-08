def test_features(data, pick_features=0):
    # Exctract the features and y_labels we'll be using for training
    columns_to_drop = ['sentence', 'is_train', 'y_label']
    
    if pick_features == 0:
        # Use all the features, just drop the sentence, is_train and y_label
        features = data.drop(columns_to_drop, axis=1).columns
        
    elif pick_features == 1:
        # Look only at the dependency tags
        columns_to_drop += ent_labels + tag_labels
        features = data.drop(columns_to_drop, axis=1).columns
        
    elif pick_features == 2:
        # Look only at the entity tags
        columns_to_drop += dep_labels + tag_labels
        features = data.drop(columns_to_drop, axis=1).columns
        
    elif pick_features == 3:
        # Look only at the POS tags
        columns_to_drop += dep_labels + ent_labels
        features = data.drop(columns_to_drop, axis=1).columns
        
    elif pick_features == 4:
        # Keep only the most important
        features = data[most_important].columns
        
    elif pick_features == 5:
        # Keep only POS and entities
        columns_to_drop += dep_labels
        features = data.drop(columns_to_drop, axis=1).columns
        
    else:
        raise("Invalid choice, pick between 0 and 4 inclusive.")
        return
    
    return features

#==============================================================================
# Dependency Parsing
#==============================================================================
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
    'TRAILING_DEP': 0 }
dep_labels = list(dep_dict.keys())

#==============================================================================
# Entity Recognition
#==============================================================================
entity_dict = {
    'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
    'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
    'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
    'ORDINAL': 0, 'CARDINAL': 0 }
ent_labels = list(entity_dict.keys())

#==============================================================================
# Part of Speech tags
#==============================================================================
tag_dict = {
    '-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0, 
    '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
    'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0, 
    'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0, 
    'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 
    'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 
    'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
    'OOV': 0, 'TRAILING_SPACE': 0 }
tag_labels = list(tag_dict.keys())
# NNP  => Proper Noun, Singular (Names, Countries, Cities etc.)
# PRP  => Pronoun, Personal (he, she, I, me etc.)
# NORP => Entity, Nationalities or religious or political groups.
# NNS  => Noun, Plural (cats, houses, potatoes etc.)
# VBD  => Verb, Past Tense (was, been, wrote etc.)

#==============================================================================
# Least important
#==============================================================================
least_important = ['NIL', 'HVS', 'GW',
                   'BES', '#', '""', 'ADD', 'WP$', 'LS', 'XX', 'NFP', 'SYM',
                   'AFX', 'UH', 'RBS', '$', 'PDT', 'WP', 'WRB', 'FW',
                   'WORK_OF_ART', 'LAW', 'LANGUAGE', 'MONEY', 'QUANTITY']
# Description of least important features
# NIL - missing tag
# HVS - forms of have
# GW  - additional word in multi-word expression
# BES - auxiliary 'be'
# #   - symbol, number sign
# ""  - closing quotation mark
# ADD - email
# WP$ - wh-pronoun, possesive
# LS  - list item marker
# XX  - uknown
# NFP - superfluous punctuation
# SYM - symbol
# AFX - affix
# UH  - interjection
# RBS - adverb, superlative
# $   - symbol, currency
# PDT - predeterminer
# WP  - wh-pronoun, personal
# WRB - wh-adverb
# FW  - foreign word

#==============================================================================
# Most important
#==============================================================================
most_important = ['pobj', 'acomp', 'punct', 'nsubj', 'amod', 'det', 'compound', 'advmod', 'TRAILING_DEP', 'dobj', 'nsubjpass', 'conj', 'aux',
                  'NORP', 'DATE', 'PERSON', 'GPE', 'ORG', 'EVENT', 'CARDINAL', 'LOC',
                  'NNP', 'NNS', 'PRP', 'VBD', 'HYPH', 'IN', 'VBN', 'NN', 'DT', 'NNPS', 'RB', 'JJ', 'CC', ',', 'CD']

