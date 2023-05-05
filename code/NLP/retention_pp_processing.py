from types_pp_processing import cleanHtml
import spacy
nlp = spacy.load('en_core_web_sm')
def retention_process(txt):
    text = ""
    result = cleanHtml(txt)
    for sen in result:
        text += sen
    time = ""
    doc = nlp(text)
    flag = 0
    for token in doc:
        if flag == 1:
            if token.text == "year" or token.text == "month" or token.text == "week" or token.text == "day" or token.text == "hour":
                time += " " + token.text
                break
            else:
                flag = 0
        if token.pos_ == "NUM":
            flag = 1
            time = token.text
    if time == "":
        time = "The privacy policy does not specify how long the data will be retained"
    return time,text
