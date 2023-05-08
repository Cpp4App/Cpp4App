import multiprocessing
import glob
import time
import json
from tqdm import tqdm
from os.path import join as pjoin, exists
import cv2
import os
import shutil

from detect_merge.merge import reassign_ids
import detect_compo.ip_region_proposal as ip
from detect_merge.Element import Element
import detect_compo.lib_ip.ip_preprocessing as pre
import torch
import numpy as np
from torchvision import models
from torch import nn
import pandas as pd
import csv
import re
import openai
import random

# ----------------- load pre-trained classification model ----------------

model = models.resnet18().to('cpu')
in_feature_num = model.fc.in_features
model.fc = nn.Linear(in_feature_num, 99)
model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=(3, 3), stride=(2, 2),
                        bias=False)

PATH = "./model/model-99-resnet18.pkl"
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

model.eval()

# ----------------- end loading ------------------------------------------

keyword_list = {'Name':['name', 'first name', 'last name', 'full name', 'real name', 'surname', 'family name', 'given name'],
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

label_dic ={'72':'Location', '42':'Photos', '77':'Social media', '91':'Voices', '6':'Email', '89':'Social media', '40':'Location', '43':'Phone', '82':'Photos',
                                                                        '3':'Contacts', '68':'Contacts', '49':'Profile', '56':'Photos'}

# def get_data_type(sentence, keywords, use_gpt=True):
#
#     sent_data_type = "others"
#
#     if use_gpt:
#         openai.api_key = os.environ["OPENAI_API_KEY"]
#
#         prompt = f"Is this piece of texts \"{sentence}\" related to any following privacy information data types? Or not relevant to any of them? ONLY answer the data type or \"not relevant\". ONLY use following data type list. Data types and their Description:\n" \
#                  f"Name: How a user refers to themselves," \
#                  f" Birthday: A user’s birthday," \
#                  f" Address: A user’s address," \
#                  f" Phone: A user’s phone number," \
#                  f" Email: A user’s email address," \
#                  f" Contacts: A user’s contact information, or the access to the contact permission," \
#                  f" Location: A user’s location information, or the access to the location permission," \
#                  f" Photos: A user’s photos, videos, or the access to the camera permission," \
#                  f" Voices: A user’s voices, recordings, or the access to the microphone permission," \
#                  f" Financial Info: Information about a user’s financial accounts, purchases, or transactions," \
#                  f" Profile: A user’s account information," \
#                  f"Social Media: A user's social media information, or the access to social media accounts"
#
#         numbered_sentences = '\n'.join([f"{i+1}: [{s.text_content.lower()}]" for i, s in enumerate(sentence)])
#         # prompt = f"Identify the personal data types in the following sentences, separated by commas. If a sentence doesn't contain any personal data types, write 'others'.\n{numbered_sentences}\n\nResponse:"
#         prompt = f"Are those following texts [\n{numbered_sentences}\n]\n related to any following privacy information data types? Or not relevant to any of them? ONLY answer \"not relevant\" or a data type for every text, answer in the form of [answer for text 1, answer for text 2,..., answer for text {len(sentence)}]. ONLY use following data type list. Data types and their Description:\n" \
#                  f"Name: How a user refers to themselves," \
#                  f" Birthday: A user’s birthday," \
#                  f" Address: A user’s address," \
#                  f" Phone: A user’s phone number," \
#                  f" Email: A user’s email address," \
#                  f" Contacts: A user’s contact information, or the access to the contact permission," \
#                  f" Location: A user’s location information, or the access to the location permission," \
#                  f" Photos: A user’s photos, videos, or the access to the camera permission," \
#                  f" Voices: A user’s voices, recordings, or the access to the microphone permission," \
#                  f" Financial info: Information about a user’s financial accounts, purchases, or transactions," \
#                  f" Profile: A user’s account information," \
#                  f"Social media: A user's social media information, or the access to social media accounts" \
#                  f"\n\nAnswer:"
#
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=prompt,
#             max_tokens=300,
#             n=1,
#             stop=None,
#             temperature=0,
#         )
#
#         # response = openai.ChatCompletion.create(
#         #     model="gpt-3.5-turbo",
#         #     messages=[
#         #         # {"role": "system", "content": "You are a helpful assistant."},
#         #         {"role": "user", "content": prompt}
#         #     ],
#         #     max_tokens=300,
#         #     n=1,
#         #     stop=None,
#         #     temperature=0,
#         # )
#
#         response_full_text = response.choices[0].text.strip()
#         # response_full_text = response.choices[0].message['content']
#         # for k in keywords.keys():
#         #     if k == "Financial info" or k == "Social media":
#         #         if k.lower() in response_full_text.lower():
#         #             sent_data_type = k
#         #             break
#         #     else:
#         #         words = re.split(r'\W+', response_full_text.lower())
#         #         if k.lower() in words:
#         #             sent_data_type = k
#         #             break
#
#
#         print("----------------------")
#         print("sentence: ", sentence)
#         print("prompt: ", prompt)
#         print("response: ", response_full_text)
#         print("sent_data_type: ", sent_data_type)
#
#         data_type_result = response_full_text[1:-1].split(", ")
#         print("data_type_result: ", data_type_result)
#         print(f"len of data_type_result and sentences are: {len(data_type_result)} and {len(sentence)}")
#         for i in range(len(sentence)):
#             if data_type_result[i] in keywords.keys():
#                 sentence[i].label = data_type_result[i]
#             else:
#                 sentence[i].label = "others"
#
#         print("labels: ")
#         for i in range(len(sentence)):
#             print(f"sentence \"{sentence[i].text_content}\", label is: {sentence[i].label}")
#
#     else:
#         for k in keywords.keys():
#             for w in keywords[k]:
#                 words = re.split(r'\W+', sentence.lower())
#                 if w.lower() in words:
#                     sent_data_type = k
#                     break
#             if sent_data_type != "others":
#                 break
#
#     # return sent_data_type

def get_data_type(sentence, keywords, use_gpt=True):

    sent_data_type = "others"

    if use_gpt:
        openai.api_key = os.environ["OPENAI_API_KEY"]

        prompt = f"Is this piece of texts \"{sentence}\" related to any following privacy information data types? Or not relevant to any of them? ONLY answer the data type or \"not relevant\". ONLY use following data type list. Data types and their Description:\n" \
                 f"Name: How a user refers to themselves," \
                 f" Birthday: A user’s birthday," \
                 f" Address: A user’s address," \
                 f" Phone: A user’s phone number," \
                 f" Email: A user’s email address," \
                 f" Contacts: A user’s contact information, or the access to the contact permission," \
                 f" Location: A user’s location information, or the access to the location permission," \
                 f" Photos: A user’s photos, videos, or the access to the camera permission," \
                 f" Voices: A user’s voices, recordings, or the access to the microphone permission," \
                 f" Financial Info: Information about a user’s financial accounts, purchases, or transactions," \
                 f" Profile: A user’s account information," \
                 f"Social Media: A user's social media information, or the access to social media accounts"

        # response = openai.Completion.create(
        #     engine="text-davinci-002",
        #     # engine="gpt-3.5-turbo",
        #     prompt=prompt,
        #     max_tokens=100,
        #     n=1,
        #     stop=None,
        #     temperature=0.5,
        # )

        response = openai.ChatCompletion.create(
            # engine="text-davinci-002",
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0,
        )

        # response_full_text = response.choices[0].text.strip()
        response_full_text = response.choices[0].message['content']
        for k in keywords.keys():
            if k == "Financial info" or k == "Social media":
                if k.lower() in response_full_text.lower():
                    sent_data_type = k
                    break
            else:
                words = re.split(r'\W+', response_full_text.lower())
                if k.lower() in words:
                    sent_data_type = k
                    break

        print("----------------------")
        print("sentence: ", sentence)
        print("prompt: ", prompt)
        print("response: ", response_full_text)
        print("sent_data_type: ", sent_data_type)

    else:
        for k in keywords.keys():
            for w in keywords[k]:
                words = re.split(r'\W+', sentence.lower())
                if w.lower() in words:
                    sent_data_type = k
                    break
            if sent_data_type != "others":
                break

    return sent_data_type

def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    # initialization
    # input_img_root = "C:/ANU/2022 s2/honours project/code/UIED-master/input_examples"
    # output_root = "E:/Mulong/Result/rico/rico_uied/rico_new_uied_v3"
    # data = json.load(open('E:/Mulong/Datasets/rico/instances_test.json', 'r'))
    #
    # input_imgs = [pjoin(input_img_root, img['file_name'].split('/')[-1]) for img in data['images']]
    # input_imgs = sorted(input_imgs, key=lambda x: int(x.split('/')[-1][:-4]))  # sorted by index
    #
    # key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
    #               'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-top-bar': True}

    # input_img_root = "C:/ANU/2022 s2/honours project/code/UIED-master/input_examples/"
    # output_root = "C:/ANU/2022 s2/honours project/code/UIED-master/result"

    input_img_root = "./input_examples/"
    # input_img_root = "C:/ANU/2022 s2/honours project/dataset/new/original screenshots"
    output_root = "./result_classification"
    segment_root = '../scrutinizing_alexa/txt'

    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    image_list = os.listdir(input_img_root)

    input_imgs = [input_img_root + image_name for image_name in image_list]

    key_params = {'min-grad': 4, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
                  'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-top-bar': False}

    is_ip = True
    is_clf = False
    is_ocr = True
    is_merge = True
    is_classification = True

    # Load deep learning models in advance
    compo_classifier = None
    if is_ip and is_clf:
        compo_classifier = {}
        from cnn.CNN import CNN
        # compo_classifier['Image'] = CNN('Image')
        compo_classifier['Elements'] = CNN('Elements')
        # compo_classifier['Noise'] = CNN('Noise')
    ocr_model = None
    if is_ocr:
        import detect_text.text_detection as text

    # set the range of target inputs' indices
    num = 0
    # start_index = 30800  # 61728
    # end_index = 100000

    img_time_cost_all = []
    ocr_time_cost_all = []
    ic_time_cost_all = []
    ts_time_cost_all = []
    cd_time_cost_all = []

    # # Open the output CSV file for writing
    # with open(output_root+'/output.xlsx', 'w') as f:
    #     # Write the header row
    #     f.write('screenshot,id,text\n')

    # # Create an empty DataFrame with three columns
    # df = pd.DataFrame(columns=['screenshot', 'id', 'text'])
    #
    # # Write the DataFrame to an XLSX file
    # df.to_excel(output_root+'/output.xlsx', index=False)

    # with open(output_root+'/output.csv', 'w') as f:
    #     writer = csv.writer(f)
    #
    #     # Write the header row
    #     writer.writerow(['screenshot', 'id', 'text'])

    output_data = pd.DataFrame(columns=['screenshot', 'id', 'label', 'index', 'text', 'sentences'])

    for input_img in input_imgs:

        this_img_start_time = time.clock()

        resized_height = resize_height_by_longest_edge(input_img)
        index = input_img.split('/')[-1][:-4]
        # if int(index) < start_index:
        #     continue
        # if int(index) > end_index:
        #     break

        if index.split('-')[0] not in ['1']:
            continue

        if is_ocr:
            os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
            this_ocr_time_cost = text.text_detection(input_img, output_root, show=False, method='paddle')
            ocr_time_cost_all.append(this_ocr_time_cost)

        if is_ip:
            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            this_cd_time_cost = ip.compo_detection(input_img, output_root, key_params,  classifier=compo_classifier, resize_by_height=resized_height, show=False)
            cd_time_cost_all.append(this_cd_time_cost)

        if is_merge:
            import detect_merge.merge as merge

            os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
            compo_path = pjoin(output_root, 'ip', str(index) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(index) + '.json')
            board_merge, components_merge = merge.merge(input_img, compo_path, ocr_path, pjoin(output_root, 'merge'), is_remove_top_bar=key_params['remove-top-bar'], show=False)
            # ic_time_cost_all.append(this_ic_time_cost)
            # ts_time_cost_all.append(this_ts_time_cost)

        if is_classification:

            os.makedirs(pjoin(output_root, 'classification'), exist_ok=True)
            merge_path = pjoin(output_root, 'merge', str(index) + '.json')
            merge_json = json.load(open(merge_path, 'r'))
            os.makedirs(pjoin(output_root, 'classification', 'GUI'), exist_ok=True)

            # ratio = 2972/800

            # load text and non-text compo
            ele_id = 0
            compos = []
            texts = []
            elements = []

            for compo in merge_json['compos']:
                if compo['class'] == 'Text':
                    element = Element(ele_id,
                                      (compo["position"]['column_min'], compo["position"]['row_min'],
                                       compo["position"]['column_max'], compo["position"]['row_max']),
                                      'Text', text_content=compo['text_content'])
                    texts.append(element)
                    ele_id += 1
                else:
                    element = Element(ele_id,
                                      (compo["position"]['column_min'], compo["position"]['row_min'],
                                       compo["position"]['column_max'], compo["position"]['row_max']),
                                      compo['class'])
                    compos.append(element)
                    ele_id += 1

            resize_by_height = 800
            org, grey = pre.read_img(input_img, resize_by_height)

            grey = grey.astype('float32')
            grey = grey / 255

            # normalization
            grey = (grey - grey.mean()) / grey.std()

            # grey = (grey - 0.5331238) / 0.33663777

            # --------- classification ----------

            classification_start_time = time.clock()

            for compo in compos:

                comp_grey = grey[compo.row_min:compo.row_max, compo.col_min:compo.col_max]

                # cv2.imshow("comp_grey", comp_grey)
                # cv2.waitKey(0)

                # print("comp_crop: ", comp_crop)
                # comp_crop = comp_grey.reshape(1, 1, 32, 32)
                comp_crop = cv2.resize(comp_grey, (32, 32))
                # print("comp_crop: ", comp_crop)

                # cv2.imshow("comp_crop", comp_crop)
                # cv2.waitKey(0)

                comp_crop = comp_crop.reshape(1, 1, 32, 32)

                comp_tensor = torch.tensor(comp_crop)
                comp_tensor = comp_tensor.permute(0, 1, 3, 2)
                # print("comp_tensor: ", comp_tensor)
                # comp_float = comp_tensor.to(torch.float32)
                # print("comp_float: ", comp_float)
                # pred_label = model(comp_float)
                pred_label = model(comp_tensor)
                # print("output: ", pred_label)
                # print("label: ", np.argmax(pred_label.cpu().data.numpy(), axis=1))

                # if np.argmax(pred_label.cpu().data.numpy(), axis=1) in [72.0, 42.0, 77.0, 91.0, 6.0, 89.0, 40.0, 43.0, 82.0,
                #                                                         3.0, 68.0, 49.0, 56.0, 89.0]:
                #     elements.append(compo)

                if str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0]) in label_dic.keys():
                    compo.label = label_dic[str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0])]
                    elements.append(compo)
                else:
                    compo.label = str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0])
                    # compo.label = "others"

                # elements.append(compo)

                # cv2.imshow("comp_crop", comp_crop.reshape(32, 32))
                # cv2.waitKey(0)

            time_cost_ic = time.clock() - classification_start_time
            print("time cost for icon classification: %2.2f s" % time_cost_ic)
            ic_time_cost_all.append(time_cost_ic)

            # --------- end classification ----------

            text_selection_time = time.clock()

            # retries = 10
            # for i in range(retries):
            #     try:
            #         get_data_type(texts, keyword_list, use_gpt=True)
            #         break
            #     except openai.error.RateLimitError as e:
            #         if "overloaded" in str(e):
            #             # Exponential backoff with jitter
            #             sleep_time = 2 * (2 ** i) + random.uniform(0, 0.1)
            #             time.sleep(sleep_time)
            #         else:
            #             raise
            #     except Exception as e:
            #         raise

            for this_text in texts:
                # found_flag = 0
                #
                # for key in keyword_list:
                #     for w in keyword_list[key]:
                #         words = re.split(r'\W+', this_text.text_content.lower())
                #         if w.lower() in words:
                #             this_text.label = key
                #             elements.append(this_text)
                #             found_flag = 1
                #             break
                #
                # if found_flag == 0:
                #     this_text.label = 'others'

                retries = 10
                for i in range(retries):
                    try:
                        text_label = get_data_type(this_text.text_content.lower(), keyword_list, use_gpt=True)
                        break
                    except openai.error.RateLimitError as e:
                        if "overloaded" in str(e):
                            # Exponential backoff with jitter
                            sleep_time = 2 * (2 ** i) + random.uniform(0, 0.1)
                            time.sleep(sleep_time)
                        else:
                            raise
                    except Exception as e:
                        raise

                this_text.label = text_label
                # # this_text.label = get_data_type(this_text.text_content.lower(), keyword_list, use_gpt=True)
                if this_text.label != "others":
                    elements.append(this_text)

                # elements.append(this_text)

            time_cost_ts = time.clock() - text_selection_time
            print("time cost for text selection: %2.2f s" % time_cost_ts)
            ts_time_cost_all.append(time_cost_ts)

            # ---------- end -------------------------------

            show = False
            wait_key = 0

            reassign_ids(elements)
            board = merge.show_elements(org, elements, show=show, win_name='elements after merging', wait_key=wait_key)
            board_one_element = merge.show_one_element(org, elements, show=show, win_name='elements after merging', wait_key=wait_key)

            classification_root = pjoin(output_root, 'classification')

            # save all merged elements, clips and blank background
            name = input_img.replace('\\', '/').split('/')[-1][:-4]
            components = merge.save_elements(pjoin(classification_root, name + '.json'), elements, org.shape)
            cv2.imwrite(pjoin(classification_root, name + '.jpg'), board)

            print("len(board_one_element): ", len(board_one_element))
            # print("")
            for i in range(len(elements)):
                e_name = str(int(elements[i].id) + 1)
                cv2.imwrite(pjoin(classification_root + '/GUI', name + '-' + e_name + '.jpg'), board_one_element[i])

            print('[Classification Completed] Input: %s Output: %s' % (input_img, pjoin(classification_root, name + '.jpg')))

            # ---------- matching result -----------

            app_id = str(index).split('-')[0]

            # with open(output_root + '/output.xlsx', 'a') as f:
            # with pd.ExcelWriter(output_root + '/output.xlsx', mode='a', engine='openpyxl') as writer:
            # with open('output.csv', 'a') as f:

            index_path = pjoin(segment_root, app_id, 'classified_sentences/keyword_index.txt')
            dict_index = {}
            if exists(index_path):
                with open(index_path, 'r') as g:
                    for line in g:
                        key, value = line.strip().split(':', 1)
                        dict_index[key] = value
            # print(dict_index.keys())

            for item in elements:
                complete_path = pjoin(segment_root, app_id, 'classified_sentences', item.label + '.txt')
                print("complete_path: ", complete_path)

                if exists(complete_path):
                    # print("exist!!!!!!!!!!!!!!")
                    with open(complete_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    # Replace line breaks with spaces and strip any extra whitespace
                    this_text = ' '.join(content.splitlines()).strip()

                    # If there's a match, write it to the CSV file
                    # f.write(f'{app_id},{item.id},{this_text}\n')

                    # new_df = pd.DataFrame({'screenshot': [str(index)], 'id': [item.id], 'text': [this_text]})
                    # new_df.to_excel(writer, index=False, header=False)

                    # writer = csv.writer(f)
                    # writer.writerow([str(index), item.id, this_text])
                    # f.flush()
                    # # print(new_df)

                    lines = content.splitlines()
                    non_empty_lines = [line for line in lines if line.strip() != ""]
                    for i in range(len(non_empty_lines)):
                        if non_empty_lines[i][0].isalpha():
                            non_empty_lines[i] = non_empty_lines[i][0].upper() + non_empty_lines[i][1:]
                    # print("non_empty_lines: ", non_empty_lines)
                    output_data = output_data.append({'screenshot': 's' + str(index), 'id': item.id + 1, 'label': item.label, 'index': dict_index[item.label], 'text': this_text, 'sentences': non_empty_lines}, ignore_index=True)

                else:
                    # f.write(f'{app_id},{item.id},{"No information!"}\n')
                    # df = pd.DataFrame({'screenshot': [str(index)], 'id': [item.id], 'text': ["No information!"]})
                    # df.to_excel(writer, index=False, header=False)
                    # writer = csv.writer(f)
                    # writer.writerow([str(index), item.id, "No information!"])
                    # f.flush()
                    output_data = output_data.append({'screenshot': 's' + str(index), 'id': item.id + 1, 'label': item.label, 'index': "None", 'text': "No information!", 'sentences': "None"},
                                                     ignore_index=True)

        this_img_time_cost = time.clock() - this_img_start_time
        img_time_cost_all.append(this_img_time_cost)
        print("time cost for this image: %2.2f s" % this_img_time_cost)

        num += 1

    output_data.to_csv(output_root + '/output.csv', index=False, mode='w')

    avg_ocr_time_cost = sum(ocr_time_cost_all) / len(ocr_time_cost_all)
    avg_cd_time_cost = sum(cd_time_cost_all) / len(cd_time_cost_all)
    avg_ic_time_cost = sum(ic_time_cost_all) / len(ic_time_cost_all)
    avg_ts_time_cost = sum(ts_time_cost_all) / len(ts_time_cost_all)
    avg_time_cost = sum(img_time_cost_all)/len(img_time_cost_all)
    print("average text extraction time cost for this app: %2.2f s" % avg_ocr_time_cost)
    print("average widget detection time cost for this app: %2.2f s" % avg_cd_time_cost)
    print("average icon classification time cost for this app: %2.2f s" % avg_ic_time_cost)
    print("average text selection processing time cost for this app: %2.2f s" % avg_ts_time_cost)
    print("average screenshot processing time cost for this app: %2.2f s" % avg_time_cost)
