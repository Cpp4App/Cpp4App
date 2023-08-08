import re

import bs4

from paragraph_bayesian import clf,tf
from bs4 import BeautifulSoup

mark_txt = {'0':"/data_types.txt",
            '1':"/data_types.txt",
            '2':"/personal_information_type.txt",
            '3':"/share_information.txt",
            '4':"/protect_information.txt",
            '5':"/advertising.txt",
            '6':"/user_right.txt",
            '7':"/children.txt",
            '8':"/region.txt",
            '9':"/update.txt",
            '10':"/way_to_collect.txt",
            '11':"/provider.txt",
            '12':"/data_retention.txt",
            '13':"/data_types.txt",
            '14':"/thrid_party.txt",
            '15':"/data_types.txt"}

def write_text(title_list, pathName):
    type = 0
    security = 0
    right = 0
    specialGroup = 0
    specialArea = 0
    update = 0
    retention = 0
    useData = 0
    clean_title_list = []
    for title in title_list:
        if title.text != "•":
            clean_title_list.append(title)

    # print("title list:"+str(clean_title_list))

    lastMark = ""
    for title in clean_title_list:
        title_Str = re.sub(r'\s+', ' ',str(title))
        title_Str = re.sub(r'<[^<]+?>', '', title_Str).replace('\n','').strip()
        if title is None:
            continue
        try:
            mark = clf.predict(tf.transform([title_Str]))

        except Exception as e:
            continue
        # print(mark)
        if mark == "1":
            type = 1
        if mark == "4":
            security = 1
        if mark == "6":
            right = 1
        if mark == "13":
            useData = 1
        if mark == "8":
            specialArea = 1
        if mark == "9":
            update = 1
        if mark == "12":
            retention = 1

        if mark == "7":
            specialGroup = 1
        if mark == "0":
            if lastMark != "":
                mark = lastMark
        lastMark = mark
        for sibling in title.next_elements:
            # print("sibling", sibling)

            # if len(str(sibling).split(' ')) < 5:
            #     continue
            try:
                if clean_title_list[clean_title_list.index(title) + 1] == sibling:

                    break
            except Exception:
                continue
            # if isinstance(sibling, bs4.element.Tag):
            #
            #     continue
            if str(sibling) == '\n':

                continue
            if sibling == title.string:

                continue

            if clean_title_list.index(title) == len(clean_title_list) - 1:

                with open('./txt/'+pathName[:-5]+mark_txt.get(mark[0]),"a",encoding='utf-8') as f:

                    if sibling.name is None or (sibling.name != 'li' and sibling.name != 'p' and sibling.name != 'br' and isinstance(sibling, bs4.element.Tag)):
                        continue
                    if sibling.name == 'li':

                        if sibling.find_previous('p'):

                            # p_text = sibling.find_previous('p').text.strip()
                            parent = ' '.join(sibling.find_previous('p').text.split())
                            text = ' '.join(sibling.get_text().split())
                            currentSibing = f"{parent} {text}"
                            # if currentSibing[-1].isalpha() or currentSibing[-1] == ")":
                            #     currentSibing = currentSibing + "."
                            # g.write(currentSibing)
                            # print("Found ul after a p tag with text:", parent)
                        else:
                            # currentSibing = str(sibling)
                            currentSibing = ' '.join(sibling.get_text().split())
                    else:
                        # currentSibing = str(sibling)
                        currentSibing = ' '.join(sibling.get_text().split())
                    # currentSibing = str(sibling)
                    if len(currentSibing) != 0:
                        if currentSibing[-1].isalpha() or currentSibing[-1] == ")":
                            currentSibing = currentSibing + "."
                        elif currentSibing[-1] == ";" or currentSibing[-1] == ":" or currentSibing[-1] == ",":
                            currentSibing = currentSibing[:-1]
                            currentSibing = currentSibing + "."

                        f.write(currentSibing)
                        f.write("\n")
                        f.close()

            else:

                with open('./txt/'+pathName[:-5]+mark_txt.get(mark[0]),"a",encoding='utf-8') as g:

                    if sibling.name is None or (sibling.name != 'li' and sibling.name != 'p' and sibling.name != 'br' and isinstance(sibling, bs4.element.Tag)):
                        continue
                    if sibling.name == 'li':

                        if sibling.find_previous('p'):

                            # p_text = sibling.find_previous('p').text.strip()
                            parent = ' '.join(sibling.find_previous('p').text.split())
                            text = ' '.join(sibling.get_text().split())
                            currentSibing = f"{parent} {text}"
                            # if currentSibing[-1].isalpha() or currentSibing[-1] == ")":
                            #     currentSibing = currentSibing + "."
                            # g.write(currentSibing)
                            # print("Found ul after a p tag with text:", parent)
                        else:
                            # currentSibing = str(sibling)
                            currentSibing = ' '.join(sibling.get_text().split())
                    else:
                        # currentSibing = str(sibling)
                        currentSibing = ' '.join(sibling.get_text().split())
                    # currentSibing = str(sibling)
                    if len(currentSibing) != 0:
                        if currentSibing[-1].isalpha() or currentSibing[-1] == ")":
                            currentSibing = currentSibing + "."
                        elif currentSibing[-1] == ";" or currentSibing[-1] == ":" or currentSibing[-1] == ",":
                            currentSibing = currentSibing[:-1]
                            currentSibing = currentSibing + "."
                        g.write(currentSibing)
                        g.write("\n")
                        g.close()

    return type,security,right,specialArea,specialGroup,update,retention,useData

def write_text_without_label(text, pathName):
    with open('./txt/' + pathName[:-5] + '/data_types.txt', "a", encoding='utf-8') as f:
        currentSibing = str(text)
        # print("currentSibing", currentSibing)
        if currentSibing[-1].isalpha() or currentSibing[-1] == ")":
            currentSibing = currentSibing + "."
        elif currentSibing[-1] == ";":
            currentSibing[-1] = "."
        f.write(currentSibing)
        f.close()

def removeUnneccessaryElements(soup):
    for script in soup(["script", "style", "nav", "footer", "header", "img", "option", "select", "head", "button"]):
        script.extract()  # rip it out
    for div in soup.find_all("div", {'class': 'footer'}):
        div.decompose()
    for div in soup.find_all("div", {'class': re.compile(r"sidebar")}):
        div.decompose()
    for div in soup.find_all("div", {'data-testid': re.compile(r"ax-navigation-menubar")}):
        div.decompose()
    for div in soup.find_all("div", {'class': re.compile(r"menu")}):
        div.decompose()
    for li in soup.find_all("li", {'class': re.compile(r"menu")}):
        li.decompose()
    for p in soup.find_all("p", {'class': re.compile(r"heading")}):
        p.decompose()
    for p in soup.find_all("p", {'class': re.compile(r"fw-bold")}):
        p.decompose()
    for ul in soup.find_all("ul", {'class': re.compile(r"menu")}):
        ul.decompose()
    for div in soup.find_all("div", {'class': re.compile(r"header")}):
        div.decompose()
    for div in soup.find_all("div", {'data-referrer': re.compile(r"page_footer")}):
        div.decompose()
    for div in soup.find_all("div", {'id': 'footer'}):
        div.decompose()
    for div in soup.find_all("div", {'id': re.compile(r"sidebar")}):
        div.decompose()
    for div in soup.find_all("div", {'id': re.compile(r"menu")}):
        div.decompose()
    for li in soup.find_all("li", {'id': re.compile(r"menu")}):
        li.decompose()
    for ul in soup.find_all("ul", {'id': re.compile(r"menu")}):
        ul.decompose()
    for div in soup.find_all("div", {'id': re.compile(r"header")}):
        div.decompose()
    for div in soup.find_all("div", {'id': re.compile(r"breadcrumbs")}):
        div.decompose()
    for div in soup.find_all("div", {'id': re.compile(r"instagram")}):
        div.decompose()
    for div in soup.find_all("div", {'role': re.compile(r"navigation")}):
        div.decompose()
    for div in soup.find_all("div", {'role': re.compile(r"banner")}):
        div.decompose()
    for div in soup.find_all("div", {'role': re.compile(r"button")}):
        div.decompose()
    for div in soup.find_all("ul", {'role': re.compile(r"navigation")}):
        div.decompose()

def makeCoarseSegments(soup):
    segments = []
    for p in soup.find_all("p"):
        if p.find_next() is not None:
            if p.find_next().name != "ul":
                # segments.append(' '.join(p.get_text().split()))
                text = ' '.join(p.get_text().split())

                if len(text) != 0:
                    if text[-1].isalpha() or text[-1] == ")":
                        text = text + "."
                    elif text[-1] == ";" or text[-1] == ":" or text[-1] == ",":
                        text = text[:-1]
                        text = text + "."

                segments.append(text)

    listSplitter = []

    for ul in soup.find_all("ul"):
        if ul.find_previous('p') is not None:
            parent = ' '.join(ul.find_previous('p').text.split())
            for element in ul.findChildren('li'):
                text = ' '.join(element.get_text().split())
                listElement = f"{parent} {text}"

                if len(listElement) != 0:
                    if listElement[-1].isalpha() or listElement[-1] == ")":
                        listElement = listElement + "."
                    elif listElement[-1] == ";" or listElement[-1] == ":" or listElement[-1] == ",":
                        listElement = listElement[:-1]
                        listElement = listElement + "."

                segments.append(listElement)
        else:
            for element in ul.findChildren('li'):
                text = ' '.join(element.get_text().split())

                if len(text) != 0:
                    if text[-1].isalpha() or text[-1] == ")":
                        text = text + "."
                    elif text[-1] == ";" or text[-1] == ":" or text[-1] == ",":
                        text = text[:-1]
                        text = text + "."

                segments.append(text)

    # if not segments:
    #     text = soup.getText().replace('\n', '').replace('↵', '')
    #     result = useAlgorithm(text)
    # else:
    #     # text = " ".join(segments)
    #     # print("TEXT??", text)
    #     print("SEGMENTS??", segments)
    #     result = segments
    result = segments
    return result
