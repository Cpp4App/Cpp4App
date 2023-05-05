import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize,pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def tokenize(sentence):

    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)
    token_words = pos_tag(token_words)
    return token_words


wordnet_lematizer = WordNetLemmatizer()
def stem(token_words):

    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='r')
        else:
            word_lematizer =  wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer

def delete_invalid_word(token_words):
    valid_word = []
    for word in token_words:
        if len(wordnet.synsets(word)) > 0:
            valid_word.append(word)
    return valid_word

sr = stopwords.words('english')
sr.append("limited")
sr.append("additionnaly")
sr.append("e.g")
sr.remove("other")
sr.remove("than")
sr.remove("not")
sr.remove("you")
sr.remove("and")
sr2 = stopwords.words('english')
def delete_stopwords(token_words):

    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words

def delete_stopwords2(token_words):

    cleaned_words = [word for word in token_words if word not in sr2]
    return cleaned_words

def delete_adjwords(token_words):

    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words


def is_number(s):

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
characters_title = [' ','.',',','|' , ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','...','^','{','}']
characters = [' ','|' , ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','...','^','{','}']
characters_proposal = [' ','|' , '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','...','^','{','}']
def delete_characters(token_words):

    words_list = [word for word in token_words if word not in characters]
    return words_list
def delete_characters_proposal(token_words):

    words_list = [word for word in token_words if word not in characters_proposal and not is_number(word)]
    return words_list

def delete_characters_title(token_words):

    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list


def to_lower(token_words):

    words_lists = [x.lower() for x in token_words]
    return words_lists

def pre_process_title(text):

    token_words = tokenize(text)

    token_words = stem(token_words)

    token_words = delete_invalid_word(token_words)

    token_words = delete_characters_title(token_words)

    token_words = to_lower(token_words)

    return ' '.join(token_words)

def pre_process(text):

    token_words = tokenize(text)

    token_words = stem(token_words)

    token_words = delete_stopwords(token_words)

    token_words = delete_characters(token_words)

    token_words = to_lower(token_words)

    return ' '.join(token_words)

def pre_process_type(text):

    token_words = tokenize(text)

    token_words = stem(token_words)

    token_words = delete_stopwords2(token_words)

    token_words = delete_characters(token_words)

    token_words = to_lower(token_words)

    return ' '.join(token_words)


def pre_process_proposal(text):

    token_words = tokenize(text)

    token_words = stem(token_words)

    token_words = delete_stopwords(token_words)

    token_words = delete_characters_proposal(token_words)

    token_words = to_lower(token_words)

    return ' '.join(token_words)

def pre_process_list(text):

    token_words = tokenize(text)

    token_words = stem(token_words)

    token_words = delete_stopwords(token_words)

    token_words = delete_characters(token_words)

    token_words = to_lower(token_words)

    return token_words

def pre_process_stop(text):

    token_words = tokenize(text)

    token_words = stem(token_words)

    token_words = delete_characters(token_words)

    token_words = to_lower(token_words)

    text = ' '.join(token_words)
    final_text = text.split(".")
    return final_text
