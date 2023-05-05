from text_preprocessing import pre_process_list

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
# intend
def process_specialGroup(txt):
    specialGroup = ""
    age = ""
    rule = ""
    childUse = 0
    with open(txt, encoding='utf-8') as file_obj:
        for line in file_obj:
            specialGroup += line
    result = pre_process_list(specialGroup)

    flag = 0
    for word in result:
        # print("word in result: ", word)
        if word == "direct" or word == "intend" or word == "address":
            childUse = 1
        if is_number(word):
            if word != age and age == "":
                age = word
        if word == "coppa":
            if rule != word:
                rule = "COPPA"
                flag = 1
        if word == "gdpr":
            if rule != word:
                rule = "GDPR"
                flag = 1
    if flag == 0:
        rule += "The privacy policy does not specify what rules to follow"
    if age =="":
        age = "The privacy policy does not mention the age of the child"
    return age , rule,childUse,specialGroup
