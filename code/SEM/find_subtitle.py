import csv
import os
import bs4


def find_title_Label(path):
    a = 0
    soup = bs4.BeautifulSoup(open(path,encoding='utf-8'), features="html.parser")
    all_list = ["","","","","","",""]
    list_index = ['h1','h2','h3','h4','h5','strong','b']
    h1_list = soup.find_all('h1')
    if len(h1_list) <= 2:
        h1_list = None
    try:
        for h1 in h1_list:
            all_list[0] += h1.text

    except Exception:
        a = 1
    h2_list = soup.find_all('h2')
    if len(h2_list) <= 2:
        h2_list = None
    try:
        for h2 in h2_list:
            all_list[1] += h2.text
    except Exception:
        a = 1
    h3_list = soup.find_all('h3')
    if len(h3_list) <= 2:
        h3_list = None
    try:
        for h3 in h3_list:
            all_list[2] += h3.text
    except Exception:
        a = 1
    h4_list = soup.find_all('h4')
    if len(h4_list) <= 2:
        h4_list = None
    try:
        for h4 in h4_list:
            all_list[3] += h4.text
    except Exception:
        a = 1
    h5_list = soup.find_all('h5')
    if len(h5_list) <= 2:
        h5_list = None
    try:
        for h5 in h5_list:
            all_list[4] += h5.text
    except Exception:
        a = 1
    strong_list = soup.find_all('strong')
    if len(strong_list) <= 2:
        strong_list = None
    try:
        for st in strong_list:
            all_list[5] += st.text
    except Exception:
        a = 1
    b_list = soup.find_all('b')
    if len(b_list) <= 2:
        b_list = None
    try:
        for b in b_list:
            all_list[6] += b.text
    except Exception:
        a = 1
    long = 0
    maxLongList = None
    for list in all_list:
        if list == None:
            continue
        clean_list = list.lower()

        if "information" in clean_list and "collect" in clean_list:

            return list_index[all_list.index(list)]
        if "information" in clean_list and "use" in clean_list:

            return list_index[all_list.index(list)]
        if "change" in clean_list and "data" in clean_list:

            return list_index[all_list.index(list)]
        if len(list) > long:
            long = len(list)
            maxLongList = list
    if maxLongList == None:
        return "TitleError"

    return list_index[all_list.index(maxLongList)]

def find_title_Label_with_html(file):
    a = 0
    soup = bs4.BeautifulSoup(file, features="html.parser")
    all_list = ["","","","","","",""]
    list_index = ['h1','h2','h3','h4','h5','strong','b']
    h1_list = soup.find_all('h1')
    if len(h1_list) <= 2:
        h1_list = None
    try:
        for h1 in h1_list:
            all_list[0] += h1.text

    except Exception:
        a = 1
    h2_list = soup.find_all('h2')
    if len(h2_list) <= 2:
        h2_list = None
    try:
        for h2 in h2_list:
            all_list[1] += h2.text
    except Exception:
        a = 1
    h3_list = soup.find_all('h3')
    if len(h3_list) <= 2:
        h3_list = None
    try:
        for h3 in h3_list:
            all_list[2] += h3.text
    except Exception:
        a = 1
    h4_list = soup.find_all('h4')
    if len(h4_list) <= 2:
        h4_list = None
    try:
        for h4 in h4_list:
            all_list[3] += h4.text
    except Exception:
        a = 1
    h5_list = soup.find_all('h5')
    if len(h5_list) <= 2:
        h5_list = None
    try:
        for h5 in h5_list:
            all_list[4] += h5.text
    except Exception:
        a = 1
    strong_list = soup.find_all('strong')
    if len(strong_list) <= 2:
        strong_list = None
    try:
        for st in strong_list:
            all_list[5] += st.text
    except Exception:
        a = 1
    b_list = soup.find_all('b')
    if len(b_list) <= 2:
        b_list = None
    try:
        for b in b_list:
            all_list[6] += b.text
    except Exception:
        a = 1
    long = 0
    maxLongList = None
    for list in all_list:
        if list == None:
            continue
        clean_list = list.lower()

        if "information" in clean_list and "collect" in clean_list:

            return list_index[all_list.index(list)]
        if "information" in clean_list and "use" in clean_list:

            return list_index[all_list.index(list)]
        if "change" in clean_list and "data" in clean_list:

            return list_index[all_list.index(list)]
        if len(list) > long:
            long = len(list)
            maxLongList = list
    if maxLongList == None:
        return "TitleError"

    return list_index[all_list.index(maxLongList)]




