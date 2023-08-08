import os
import time
import shutil

from bs4 import BeautifulSoup

from find_subtitle import find_title_Label_with_html
from get_text import write_text, removeUnneccessaryElements, makeCoarseSegments
from types_pp_processing import getSentences_with_classifier


def run_single_pp(file):
    # INPUT = "../dataset/privacy_policies_html/"
    # INPUT = "./pp_example/"
    # cleaning_txt("./txt")
    # os.mkdir("./txt")
    if os.path.exists("./txt"):
        shutil.rmtree("./txt")
    os.makedirs("./txt")

    # file = os.listdir(INPUT)[0]

    segmentation_start_time = time.clock()

    pathName = "demo_pp.html"

    label = find_title_Label_with_html(file)
    print("The current file is:" + pathName)

    # if pathName != '20.html':
    #     continue

    para_start_time = time.clock()
    soup = BeautifulSoup(file, features="html.parser")
    title_list = soup.find_all(label)
    # cleaning_txt()

    if not os.path.exists('./txt/' + pathName[:-5]):
        os.mkdir('./txt/' + pathName[:-5])

    if len(title_list) == 0:
        # write_text_without_label(soup.getText(), pathName)
        removeUnneccessaryElements(soup)
        result = makeCoarseSegments(soup)
        for seg in result:
            with open('./txt/' + pathName[:-5] + '/data_types.txt', "a", encoding='utf-8') as f:
                f.write(seg)
                f.write("\n")
    else:
        write_text(title_list, pathName)
    print("Paragraph level processing time: %2.2f s" % (time.clock() - para_start_time))

    for t in title_list:
        with open('./txt/' + pathName[:-5] + '/headings.txt', "a", encoding='utf-8') as g:
            g.write(str(t))
            g.write("\n")

    # data types
    if not os.path.exists("./txt/" + pathName[:-5] + "/data_types.txt"):
        print("No information about data types!")
    else:
        sen_start_time = time.clock()
        # all_types = caculateSim("./txt/"+pathName[:-5]+"/data_types.txt")
        dict_sentences, dict_index = getSentences_with_classifier("./txt/" + pathName[:-5] + "/data_types.txt")
        print("sentence level processing time: %2.2f s" % (time.clock() - sen_start_time))

        os.makedirs("./txt/" + pathName[:-5] + "/classified_sentences")
        for key in dict_sentences:

            if dict_sentences[key] == "":
                continue
            with open('./txt/' + pathName[:-5] + "/classified_sentences/" + key + ".txt", "a",
                      encoding='utf-8') as g:
                g.write(dict_sentences[key])

        for key in dict_index:
            with open('./txt/' + pathName[:-5] + "/classified_sentences/keyword_index.txt", "a",
                      encoding='utf-8') as f:
                f.write(key + ":" + str(dict_index[key]) + "\n")

    print("time cost for segmentation: %2.2f s" % (time.clock() - segmentation_start_time))

