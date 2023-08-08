import os
import time
import shutil

from bs4 import BeautifulSoup

from find_subtitle import find_title_Label
from get_text import write_text, write_text_without_label, removeUnneccessaryElements, makeCoarseSegments
from types_pp_processing import caculateSim, getSentences, getSentences_no_classifier, getSentences_with_classifier
# from children_pp_processing import process_specialGroup
# from region_pp_processing import get_alifornia
# from retention_pp_processing import retention_process
# from clean_txt import cleaning_txt

if __name__ == '__main__':
    # INPUT = "../dataset/privacy_policies_html/"
    INPUT = "./pp_example/"
    # cleaning_txt("./txt")
    # os.mkdir("./txt")

    # if os.path.exists("./txt"):
    #     shutil.rmtree("./txt")
    # os.makedirs("./txt")

    for file in os.listdir(INPUT):

        segmentation_start_time = time.clock()

        pathName = os.path.basename(file)
        if pathName == ".DS_Store":
            continue
        path = INPUT+pathName
        label = find_title_Label(path)
        print("The current file is:" + pathName)

        para_start_time = time.clock()
        soup = BeautifulSoup(open(path,encoding='utf-8'), features="html.parser")
        title_list = soup.find_all(label)
        # cleaning_txt()

        if not os.path.exists('./txt/' + pathName[:-5]):
            os.mkdir('./txt/' + pathName[:-5])

        if len(title_list) == 0 or pathName == '20.html' or pathName == '29.html' or pathName == '25.html' or pathName == '8.html' or pathName == '27.html' or pathName == '28.html'\
                or pathName == '36.html' or pathName == '42.html':
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
        if not os.path.exists("./txt/"+pathName[:-5]+"/data_types.txt"):
            print("No information about data types!")
        else:
            sen_start_time = time.clock()
            # all_types = caculateSim("./txt/"+pathName[:-5]+"/data_types.txt")
            dict_sentences, dict_index = getSentences_with_classifier("./txt/" + pathName[:-5] + "/data_types.txt")
            print("sentence level processing time: %2.2f s" % (time.clock() - sen_start_time))

            os.makedirs("./txt/"+pathName[:-5]+"/classified_sentences")
            for key in dict_sentences:

                if dict_sentences[key] == "":
                    continue
                with open('./txt/' + pathName[:-5] + "/classified_sentences/" + key + ".txt", "a", encoding='utf-8') as g:
                    g.write(dict_sentences[key])

            for key in dict_index:
                with open('./txt/' + pathName[:-5] + "/classified_sentences/keyword_index.txt", "a", encoding='utf-8') as f:
                    f.write(key + ":" + str(dict_index[key]) + "\n")


        # #children
        # if not os.path.exists("./txt/"+pathName[:-5]+"/children.txt"):
        #     print("No information about children!")
        # else:
        #     age , rule, childUse, specialGroup = process_specialGroup("./txt/"+pathName[:-5]+"/children.txt")
        #     # print("children age is :")
        #     print("D.CHILDREN.age : " + str(age))
        #     if childUse == 1:
        #         print(" the skill’s privacy policy states that it does not collect any information from children")
        #         print("D.CHILDREN.[CTypes] = [ ]")
        #     else:
        #         # print("D.CHILDREN.[CTypes] :" + str(all_types))
        #         None
        # #region
        # if not os.path.exists("./txt/"+pathName[:-5]+"/region.txt"):
        #     print("No information about region!")
        # else:
        #     specialArea,california = get_alifornia("./txt/"+pathName[:-5]+"/region.txt")
        #     if california == 1:
        #         print("D.REGIONS.region :California")
        #         print("D.REGIONS.delete : Yes")
        #     else:
        #         print("D.REGIONS.region :No mention")
        #         print("D.REGIONS.delete : No")
        #
        # #retention
        # if not os.path.exists("./txt/"+pathName[:-5]+"/data_retention.txt"):
        #     print("No information about data retention!")
        # else:
        #     retention_time, text = retention_process("./txt/"+pathName[:-5]+"/data_retention.txt")
        #     print("D.RETENTION.period :"+ retention_time)
        #     # cleaning_txt()
        #     print("-------------------------------------------------------")

        print("time cost for segmentation: %2.2f s" % (time.clock() - segmentation_start_time))
