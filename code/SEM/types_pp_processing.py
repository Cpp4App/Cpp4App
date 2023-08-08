import csv
import re
import spacy
from bs4 import BeautifulSoup
import stanza

from nltk.corpus import stopwords, wordnet
from text_preprocessing import pre_process,pre_process_type
from sentence_bayesian import clf_type,tf
from phrase_similarity import wordnetSim3, wordnetSim_modified

def check_ngram(string):
    words = string.split()
    num_words = len(words)
    return num_words


replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'cannot'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s
# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def cleanHtml(txt):

    # only split with line
    personal_information = []
    with open(txt, encoding='utf-8') as file_obj:
        for line in file_obj:
            # if len(line.split(' ')) >= 5:
            personal_information.append(line)

    text = ''.join(personal_information)
    soup = BeautifulSoup(text, 'html.parser')
    lower = soup.get_text().lower()

    # use re
    # pattern = r'(?<!e\.g|www)\s*(?:[.;\n])\s*(?!g\.|com|org|net|edu|gov|io|info|co|au)'
    # sentence_list = re.split(pattern, lower)

    # use stanza tokenizer
    # stanza.download('en')
    stanza_nlp = stanza.Pipeline('en', processors='tokenize')
    print("Pipeline done")
    stanza_doc = stanza_nlp(lower)
    print("stanza_doc done")
    sentence_list = [sentence.text for sentence in stanza_doc.sentences]
    print("sentence_list: ", sentence_list)

    # use spacy tokenizer
    # spacy_nlp = spacy.load('en_core_web_sm')
    # spacy_doc = spacy_nlp(lower)
    # sentence_list = [sentence.text.strip() for sentence in spacy_doc.sents]
    # # print("sentence_list: ", sentence_list)

    return sentence_list

def writeSentenceFirst(sentence_list):
    f = open('personal_type.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f,dialect='unix')
    csv_writer.writerow(["mark", "sentence"])
    for sentence in sentence_list:
        csv_writer.writerow(['0', sentence])
    f.close()

def writeSentence(sentence_list):
    f = open('personal_type.csv', 'a', encoding='utf-8')
    csv_writer = csv.writer(f, dialect='unix')
    for sen in sentence_list:
        csv_writer.writerow(['0', sen])
    f.close()

def caculateSim(txt):
    information_type = ["name", "email address", "phone number", "billing","birth date", "age",'user id', "gender", "location","job title",
                        "phonebook", "sms", "income","ip","internet protocol","marital","social security number",'credit card',
                        "type browser","browser version","operate system","postal address","postcode","profile","education","occupation","student","software",
                        "driver","insurance","health","signature","province","time zone","isp","tax","device id","domain name",
                        "prior usage","cookie","web page","interact site","device information","dash cam",
                        "log data","page service visit","time spend page","time date visit","time date use service","demographic information","country","usage pattern","language","reminder","alexa notification","amazon pay"]
    sentence_list = cleanHtml(txt)
    for sen in sentence_list:
        sentence_list[sentence_list.index(sen)] = pre_process_type(sen)

    word = []
    simList = []
    for a in information_type:
        word.append(0)
    for b in information_type:
        simList.append(0)
    for sentence in sentence_list:
        for type in information_type:
            if type in sentence:
                if type == "age" or type == "interest":
                    if sentence.index(type) - 1 == " ":
                        word[information_type.index(type)] = 1
                else:
                    word[information_type.index(type)] = 1
        if clf_type.predict(tf.transform([sentence])) == "1":
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(sentence)
            chunk_list = []
            for chunk in doc.noun_chunks:
                chunk_str = str(chunk)
                if chunk_str[0] == " ":
                    chunk_str = chunk_str[1:]
                chunk_list.append(chunk_str)
            for type in information_type:
                for chunk in chunk_list:
                    if type == chunk:
                        word[information_type.index(type)] = 1
                        chunk_list.remove(chunk)
            for type in information_type:
                for chunk in chunk_list:
                    try:
                        if wordnetSim3(chunk,type) > 0.8:
                            simList[information_type.index(type)] = wordnetSim3(chunk,type)
                    except Exception:
                        pass
                        print("error")
            nowMax = 0
            for max in simList:
                if max > nowMax:
                    nowMax = max
            if nowMax != 0:
                word[simList.index(nowMax)] = 1
    return word

def getSentences(txt):

    information_type = {'Name':['name', 'first name', 'last name', 'full name', 'real name', 'surname', 'family name', 'given name'],
                        'Birthday':['birthday', 'date of birth', 'birth date', 'DOB', 'dob full birthday'],
                        'Address':['address', 'mailing address', 'physical address', 'postal address', 'billing address', 'shipping address'],
                        'Phone':['phone', 'phone number', 'mobile', 'mobile phone', 'mobile number', 'telephone', 'telephone number', 'call'],
                        'Email':['email', 'e-mail', 'email address', 'e-mail address'],
                        'Contacts':['contacts', 'phone-book', 'phone book'],
                        'Location':['location', 'locate', 'place', 'geography', 'geo', 'geo-location', 'precision location'],
                        'Camera':['camera', 'photo', 'scan', 'album', 'picture', 'gallery', 'photo library', 'storage', 'image', 'video'],
                        'Microphone':['microphone', 'voice, mic', 'speech', 'talk'],
                        'Financial':['credit card', 'company', 'companies', 'organization', 'organizations', 'pay', 'payment'],
                        'IP':['IP', 'Internet Protocol', 'IP address', 'internet protocol address'],
                        'Cookies':['cookies', 'cookie']}

    sentence_list = cleanHtml(txt)
    for sen in sentence_list:
        sentence_list[sentence_list.index(sen)] = pre_process_type(sen)

    # print("all sentences:\n")
    # for sen in sentence_list:
    #     print(sen)
    #     print("\n")

    classified_sen = {'Name':[],
                        'Birthday':[],
                        'Address':[],
                        'Phone':[],
                        'Email':[],
                        'Contacts':[],
                        'Location':[],
                        'Camera':[],
                        'Microphone':[],
                        'Financial':[],
                        'IP':[],
                        'Cookies':[]}
    # simList = []
    # for a in information_type:
    #     word.append(0)
    # for b in information_type:
    #     simList.append(0)
    for sentence in sentence_list:
        if clf_type.predict(tf.transform([sentence])) == "1":
            # print("yes sentence: "+sentence+"\n")
            for type in information_type:
                for w in information_type[type]:
                    if w in sentence:
                        if w == "geo" or w == "IP" or w == "DOB":
                            # check whether w is a part of an unrelated word
                            if sentence[sentence.index(w) - 1] == " " and sentence not in classified_sen[type]:
                                classified_sen[type].append(sentence)
                        else:
                            # check duplication
                            if sentence not in classified_sen[type]:
                                classified_sen[type].append(sentence)

    return classified_sen

def getSentences_no_classifier(txt):

    information_type = {'Name':['name', 'first name', 'last name', 'full name', 'real name', 'surname', 'family name', 'given name'],
                        'Birthday':['birthday', 'date of birth', 'birth date', 'DOB', 'dob full birthday'],
                        'Address':['address', 'mailing address', 'physical address', 'postal address', 'billing address', 'shipping address'],
                        'Phone':['phone', 'phone number', 'mobile', 'mobile phone', 'mobile number', 'telephone', 'telephone number', 'call'],
                        'Email':['email', 'e-mail', 'email address', 'e-mail address'],
                        'Contacts':['contacts', 'phone-book', 'phone book'],
                        'Location':['location', 'locate', 'place', 'geography', 'geo', 'geo-location', 'precision location'],
                        'Camera':['camera', 'photo', 'scan', 'album', 'picture', 'gallery', 'photo library', 'storage', 'image', 'video'],
                        'Microphone':['microphone', 'voice, mic', 'speech', 'talk'],
                        'Financial':['credit card', 'company', 'companies', 'organization', 'organizations', 'pay', 'payment'],
                        'IP':['IP', 'Internet Protocol', 'IP address', 'internet protocol address'],
                        'Cookies':['cookies', 'cookie']}

    sentence_list = cleanHtml(txt)
    for sen in sentence_list:
        sentence_list[sentence_list.index(sen)] = pre_process_type(sen)

    # print("all sentences:\n")
    # for sen in sentence_list:
    #     print(sen)
    #     print("\n")

    classified_sen = {'Name':[],
                        'Birthday':[],
                        'Address':[],
                        'Phone':[],
                        'Email':[],
                        'Contacts':[],
                        'Location':[],
                        'Camera':[],
                        'Microphone':[],
                        'Financial':[],
                        'IP':[],
                        'Cookies':[]}
    # simList = []
    # for a in information_type:
    #     word.append(0)
    # for b in information_type:
    #     simList.append(0)
    for sentence in sentence_list:
        # print("yes sentence: "+sentence+"\n")
        for type in information_type:
            for w in information_type[type]:
                if w in sentence:
                    if w == "geo" or w == "IP" or w == "DOB":
                        # check whether w is a part of an unrelated word
                        if sentence[sentence.index(w) - 1] == " " and sentence not in classified_sen[type]:
                            classified_sen[type].append(sentence)
                    else:
                        # check duplication
                        if sentence not in classified_sen[type]:
                            classified_sen[type].append(sentence)

    return classified_sen

def getSentences_with_classifier(txt):

    information_type = {'Name':['name', 'first name', 'last name', 'full name', 'real name', 'surname', 'family name', 'given name'],
                        'Birthday':['birthday', 'date of birth', 'birth date', 'DOB', 'dob full birthday', 'birth year'],
                        'Address':['mailing address', 'physical address', 'postal address', 'billing address', 'shipping address', 'delivery address', 'residence', 'collect address', 'personal address', 'residential address'],
                        'Phone':['phone', 'phone number', 'mobile', 'mobile phone', 'mobile number', 'telephone', 'telephone number', 'call'],
                        'Email':['email', 'e-mail', 'email address', 'e-mail address'],
                        'Contacts':['contacts', 'phone-book', 'phone book', 'phonebook', 'contact list', 'phone contacts', 'address book'],
                        'Location':['location', 'locate', 'geography', 'geo', 'geo-location', 'precision location', 'nearby'],
                        'Photos':['camera', 'photo', 'scan', 'album', 'picture', 'gallery', 'photo library', 'storage', 'image', 'video', 'scanner', 'photograph'],
                        'Voices':['microphone', 'voice', 'mic', 'speech', 'talk'],
                        'Financial info':['credit card', 'pay', 'payment', 'debit card', 'mastercard', 'wallet'],
                        'IP':['IP', 'Internet Protocol', 'IP address', 'internet protocol address'],
                        'Cookies':['cookies', 'cookie'],
                        'Social media':['facebook', 'twitter', 'socialmedia', 'social media'],
                        'Profile':['profile', 'account'],
                        'Gender':['gender']}

    sentence_list = cleanHtml(txt)

    classified_sen = {'Name': "",
                      'Birthday': "",
                      'Address': "",
                      'Phone': "",
                      'Email': "",
                      'Contacts': "",
                      'Location': "",
                      'Photos': "",
                      'Voices': "",
                      'Financial info': "",
                      'IP': "",
                      'Cookies': "",
                      'Social media': "",
                      'Profile': "",
                      'Gender': ""
                      }

    keyword_index = {'Name':[],
                        'Birthday':[],
                        'Address':[],
                        'Phone':[],
                        'Email':[],
                        'Contacts':[],
                        'Location':[],
                        'Photos':[],
                        'Voices':[],
                        'Financial info':[],
                        'IP':[],
                        'Cookies':[],
                      'Social media': [],
                      'Profile': [],
                      'Gender': []
                      }

    # simList = []
    # for a in information_type:
    #     word.append(0)
    # for b in information_type:
    #     simList.append(0)
    for sentence in sentence_list:
        # print("yes sentence: "+sentence+"\n")

        sentence = sentence.lower()

        info_found = False

        for type in information_type:
            for w in information_type[type]:

                if w.lower() in sentence:
                # if (check_ngram(w) == 1 and w.lower() in sentence.split()) or (check_ngram(w) > 1 and w.lower() in sentence):
                    if w == "geo" or w == "IP" or w == "DOB" or w == "mic":
                        if sentence[sentence.index(w.lower()) - 1] != " ":
                            continue
                    if sentence not in classified_sen[type]:

                        if re.match(r'[a-zA-Z0-9]', sentence[-1]):
                            sentence = sentence + '.'

                        # start_index = len(classified_sen[type]) + sentence.index(w.lower())
                        # end_index = start_index + len(w.lower()) - 1
                        # keyword_index[type].append([start_index, end_index])
                        # classified_sen[type] = classified_sen[type] + sentence

                        pattern = re.compile(re.escape(w.lower()))
                        for match in pattern.finditer(sentence):
                            start_index = len(classified_sen[type]) + match.start()
                            end_index = start_index + len(w) - 1
                            keyword_index[type].append([start_index, end_index])
                        # if sentence[0].isalpha():
                        #     sentence = sentence[0].upper() + sentence[1:]
                        classified_sen[type] = classified_sen[type] + sentence + '\n'
                        # sen_dict[type].append(sentence)

                    info_found = True

        if not info_found and clf_type.predict(tf.transform([sentence])) == "1":
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(sentence)
            chunk_list = []
            for chunk in doc.noun_chunks:
                chunk_str = str(chunk)
                if chunk_str[0] == " ":
                    chunk_str = chunk_str[1:]
                chunk_list.append(chunk_str)

            for type in information_type:
                found_this_type = False

                for w in information_type[type]:
                    for chunk in chunk_list:
                        if w == chunk or wordnetSim_modified(chunk, w) > 0.8:

                            if sentence not in classified_sen[type]:
                                # classified_sen[type].append(sentence)

                                if re.match(r'[a-zA-Z0-9]', sentence[-1]):
                                    sentence = sentence + '.'

                                # start_index = len(classified_sen[type]) + sentence.index(chunk)
                                # end_index = start_index + len(chunk) - 1
                                # keyword_index[type].append([start_index, end_index])
                                # classified_sen[type] = classified_sen[type] + sentence

                                pattern = re.compile(re.escape(chunk))
                                for match in pattern.finditer(sentence):
                                    start_index = len(classified_sen[type]) + match.start()
                                    end_index = start_index + len(chunk) - 1
                                    keyword_index[type].append([start_index, end_index])
                                # if sentence[0].isalpha():
                                #     sentence = sentence[0].upper() + sentence[1:]
                                classified_sen[type] = classified_sen[type] + sentence + '\n'
                                # sen_dict[type].append(sentence)

                                found_this_type = True

                    if found_this_type:
                        break

    return classified_sen, keyword_index








