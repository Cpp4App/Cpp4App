import csv
import joblib

from sklearn.naive_bayes import MultinomialNB

from text_preprocessing import pre_process_title
from sklearn.feature_extraction.text import TfidfVectorizer



def readtrain():
    with open('./training_data/title.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[0] for i in column1[1:]]
    opinion_train = [i[1] for i in column1[1:]]
    train = [content_train, opinion_train]
    return train

def segmentWord(cont):
    c = []
    for i in cont:
        clean_text = pre_process_title(i)
        c.append(clean_text)
    return c

train = readtrain()
content = segmentWord(train[1])

textMark = train[0]

train_content = content[:]
# test_content = content[450:508]
train_textMark = textMark[:]
# test_textMark = textMark[450:508]

tf = TfidfVectorizer(max_df=0.5)

train_features = tf.fit_transform(train_content)

load_pretrain_model = True

if not load_pretrain_model:

    clf = MultinomialNB(alpha=0.1)
    clf.fit(train_features,train_textMark)

    joblib.dump(clf, './model/para_model.pkl')
else:
    clf = joblib.load('./model/para_model.pkl')


