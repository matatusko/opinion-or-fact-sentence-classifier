import os, pickle
import requests, re
import time, random
import unicodedata
from bs4 import BeautifulSoup
import spacy
nlp = spacy.load('en_core_web_lg')

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

articles = ['https://en.wikipedia.org/wiki/World_War_I',
            'https://en.wikipedia.org/wiki/Industrial_Revolution',
            'https://en.wikipedia.org/wiki/October_Revolution',
            'https://en.wikipedia.org/wiki/Fermi_paradox',
            'https://en.wikipedia.org/wiki/Steam_engine',
            'https://en.wikipedia.org/wiki/Barack_Obama',
            'https://en.wikipedia.org/wiki/Amazon_(company)',
            'https://en.wikipedia.org/wiki/Netherlands',
            'https://en.wikipedia.org/wiki/Triangular_trade',
            'https://en.wikipedia.org/wiki/Song_dynasty',
            'https://en.wikipedia.org/wiki/Nanking_Massacre',
            'https://en.wikipedia.org/wiki/The_Holocaust',
            'https://en.wikipedia.org/wiki/Japan',
            'https://en.wikipedia.org/wiki/Sumo',
            'https://en.wikipedia.org/wiki/Sheshi',
            'https://en.wikipedia.org/wiki/Chickpea',
            'https://en.wikipedia.org/wiki/Treaty_of_Versailles',
            'https://en.wikipedia.org/wiki/Economics',
            'https://en.wikipedia.org/wiki/Law',
            'https://en.wikipedia.org/wiki/Thor']

def get_wiki_article(wiki_link, verbose=True):
    # Try connecting to the website and retrieve the question/answers pairs
    try:
        page_source = requests.get(wiki_link, headers=headers)
        soup = BeautifulSoup(page_source.text, 'html.parser')
        
        # Extract the title
        title = soup.find('h1', {'class' : 'firstHeading'}).contents
        title = str(title[0])
        
        # Rip out any of the unwanted HTML markup and its content
        for unwanted in soup(['script', 'style', 'li', 'ul', 'ol', 'tr', 'td', 'h1', 
                                  'h2', 'h3', 'h4', 'h5', 'h6', 'figcaption', 'dl', 
                                  'td', 'dd', 'nav', 'header', 'footer', 'table', 'th',
                                  'blockquote']):
            unwanted.extract()
        
        # Extract all the paragraphs and parse them, adding to the final article string
        wiki_article = ''
        paragraphs = soup.findAll('p')
        for paragraph in paragraphs:
            paragraph = unicodedata.normalize('NFKD', str(paragraph))
            # Get rid of any HTML markup
            paragraph = re.sub(r'</*.*?>', '', paragraph)
            # Remove the reference links in []
            paragraph = re.sub(r'\[.*?]', '', paragraph)
            try:
                if paragraph[-1] == '.' and paragraph[0].isupper():
                    wiki_article += paragraph + ' '
            except:
                pass

        if verbose:
            print('Got wiki article on topic: %s.' % title)
    # Pass if failed meanwhile, probably question is unanswered
    except:
        if verbose:
            print('Failed retrieving the article, possibly connection to the ' + \
                  'website was not successful.')
        return 'NO ARTICLE FOUND', 'NO TITLE'
        
    time.sleep(5)
    return wiki_article

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

def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()

#==============================================================================
# GET WIKI ARTICLES
#==============================================================================
all_articles = []
for article in articles:
    all_articles.append(get_wiki_article(article))
    
#==============================================================================
# RUN WIKI THROUGH SPACY METHOD
#==============================================================================
parsed_wiki_articles = []
for document in nlp.pipe(all_articles, batch_size=10000, n_threads=4):
    parsed_wiki_articles.append(divide_into_sentences(document))

assert len(parsed_wiki_articles) == len(all_articles) == len(articles)

sentences_with_features = []
for article in parsed_wiki_articles:
    for sentence in article:
        sentence_with_features = {}
        
        # Pure sentence
        sentence_with_features['sentence'] = str(sentence)
        sentence_with_features['y_label'] = 0 # 0 for fact
        # Entities found, categoried by their label
        entities_dict = number_of_specific_entities(sentence)
        sentence_with_features.update(entities_dict)
        # Parts of speech of each token in a sentence and their amount
        pos_dict = number_of_fine_grained_pos_tags(sentence)
        sentence_with_features.update(pos_dict)
        # Dependencies tags for each token in a sentence
        dep_dict = number_of_dependency_tags(sentence)
        sentence_with_features.update(dep_dict)
        
        sentences_with_features.append(sentence_with_features)
        
print('Got {} factual sentences from wikipedia.'.format(len(sentences_with_features)))
factual = sentences_with_features
#==============================================================================
# LOAD OPINIOSIS 
#==============================================================================
datapath = 'opiniosis'
datafiles = os.listdir(datapath)

opinion_sentences = []

# Run through each file in the opiniosis database and read it line by line
for file in datafiles:
    filepath = datapath + '/' + file
    with open(filepath) as file:
        content = file.readlines()
    content = [sent.strip() for sent in content]
    opinion_sentences += content

print('Got {} opinion sentences from opiniosis.'.format(len(opinion_sentences)))
    
# Clean up the sentences a bit because they are a mess and save the whole data
# in a single string for further spacy preprocessing
opinions_string = ''
for index, sentence in enumerate(opinion_sentences):
    while sentence[0] is ',' or sentence[0] is ' ':
        sentence = sentence[1:]
    sentence = re.sub(r' \.', '.', sentence)
    sentence = re.sub(r'  \.', '.', sentence)
    sentence = re.sub(r' !', '!', sentence)
    sentence = re.sub(r' , ', ', ', sentence)
    sentence = re.sub(r',  ', ', ', sentence)
    sentence = re.sub(r' \?', '\?', sentence)
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] is ':':
        sentence = sentence[:-1]
    if sentence[-1] is not '.' and sentence[-1] is not '?' and sentence[-1] is not '!':
        sentence += '.'
    opinions_string += sentence + ' '

#==============================================================================
# RUN OPINIOSIS THROUGH SPACY
#==============================================================================
parsed_opinions = nlp(opinions_string)
sentences = [sent for sent in parsed_opinions.sents]

for sentence in sentences:
    sentence_with_features = {}
    
    # Pure sentence
    sentence_with_features['sentence'] = str(sentence)
    sentence_with_features['y_label'] = 1 # 1 for opinion
    # Entities found, categoried by their label
    entities_dict = number_of_specific_entities(sentence)
    sentence_with_features.update(entities_dict)
    # Parts of speech of each token in a sentence and their amount
    pos_dict = number_of_fine_grained_pos_tags(sentence)
    sentence_with_features.update(pos_dict)
    # Dependencies tags for each token in a sentence
    dep_dict = number_of_dependency_tags(sentence)
    sentence_with_features.update(dep_dict)
    
    sentences_with_features.append(sentence_with_features)
    
print('In total got {} factual and opinionated sentences'.format(len(sentences_with_features)))

# Shuffle the whole dataset and pickle it to separate file
random.shuffle(sentences_with_features)
save_pickle(sentences_with_features, 'opinion_fact_sentences.pickle')