from detect_merge.Element import Element
import detect_compo.lib_ip.ip_preprocessing as pre
import time
import cv2
import torch
import numpy as np
from torchvision import models
from torch import nn
import pandas as pd
import re
import openai
import random
import os
from detect_merge.merge import reassign_ids
import detect_merge.merge as merge
from os.path import join as pjoin, exists

label_dic ={'72':'Location', '42':'Photos', '77':'Social media', '91':'Voices', '6':'Email', '89':'Social media', '40':'Location', '43':'Phone', '82':'Photos',
                                                                        '3':'Contacts', '68':'Contacts', '49':'Profile', '56':'Photos'}

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

        # print("----------------------")
        # print("sentence: ", sentence)
        # print("prompt: ", prompt)
        # print("response: ", response_full_text)
        # print("sent_data_type: ", sent_data_type)

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

def get_clf_model(clf_model="ResNet18", use_gpu=False):

    device = 'cpu'
    if use_gpu:
        device = 'cuda:0'

    if clf_model == "ResNet18":
        model = models.resnet18().to(device)
        in_feature_num = model.fc.in_features
        model.fc = nn.Linear(in_feature_num, 99)
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=(3, 3), stride=(2, 2),
                                bias=False)

        PATH = "./model/model-99-resnet18.pkl"
        model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))

        model.eval()
    elif clf_model == "ViT":
        model = torch.load('./model/model-99-ViT-entire.pkl', map_location=torch.device(device))
        model = model.to(device)
        model.eval()

    else:
        # replace with your own model
        None

    return model

def compo_classification(input_img, output_root, segment_root, merge_json, output_data, resize_by_height=800, clf_model="ResNet18"):
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

    org, grey = pre.read_img(input_img, resize_by_height)

    grey = grey.astype('float32')
    grey = grey / 255

    # grey = (grey - grey.mean()) / grey.std()

    # --------- classification ----------

    classification_start_time = time.clock()

    for compo in compos:

        if clf_model == "ResNet18":

            comp_grey = grey[compo.row_min:compo.row_max, compo.col_min:compo.col_max]

            comp_crop = cv2.resize(comp_grey, (32, 32))

            comp_crop = comp_crop.reshape(1, 1, 32, 32)

            comp_tensor = torch.tensor(comp_crop)
            comp_tensor = comp_tensor.permute(0, 1, 3, 2)

            model = get_clf_model(clf_model)
            pred_label = model(comp_tensor)

            if str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0]) in label_dic.keys():
                compo.label = label_dic[str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0])]
                elements.append(compo)
            else:
                compo.label = str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0])

        elif clf_model == "ViT":

            comp_grey = grey[compo.row_min:compo.row_max, compo.col_min:compo.col_max]

            comp_crop = cv2.resize(comp_grey, (224, 224))

            # Convert the image to tensor
            comp_tensor = torch.from_numpy(comp_crop)

            # Reshape and repeat along the channel dimension to convert to RGB
            comp_tensor = comp_tensor.view(1, 224, 224).repeat(3, 1, 1)

            # comp_tensor = comp_tensor.permute(0, 2, 1)

            comp_tensor = comp_tensor.unsqueeze(0)  # add a batch dimension

            model = get_clf_model(clf_model)
            # pred_label = model(comp_tensor)

            # Forward pass through the model
            with torch.no_grad():
                output = model(comp_tensor)

            # Get the predicted label
            _, predicted = torch.max(output.logits, 1)

            # print("predicted_label: ", predicted.cpu().numpy())

            if str(predicted.cpu().numpy()[0]) in label_dic.keys():
                compo.label = label_dic[str(predicted.cpu().numpy()[0])]
                elements.append(compo)
            else:
                compo.label = str(predicted.cpu().numpy()[0])

        else:
            print("clf_model has to be ResNet18 or ViT")

        # if str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0]) in label_dic.keys():
        #     compo.label = label_dic[str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0])]
        #     elements.append(compo)
        # else:
        #     compo.label = str(np.argmax(pred_label.cpu().data.numpy(), axis=1)[0])

    time_cost_ic = time.clock() - classification_start_time
    print("time cost for icon classification: %2.2f s" % time_cost_ic)
    # ic_time_cost_all.append(time_cost_ic)

    # --------- end classification ----------

    text_selection_time = time.clock()

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

        retries = 5

        # for i in range(retries):
        #     try:
        #         text_label = get_data_type(this_text.text_content.lower(), keyword_list, use_gpt=True)
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

        for i in range(retries):
            try:
                text_label = get_data_type(this_text.text_content.lower(), keyword_list, use_gpt=True)
                break
            except openai.error.RateLimitError as e:
                if "overloaded" in str(e):
                    # Exponential backoff with jitter
                    sleep_time = 2 * (2 ** i) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
            except Exception as e:
                # If you wish, you can print or log the exception details here without raising it
                print(e)
        else:
            # This part will be executed if the for loop doesn't hit 'break'
            text_label = get_data_type(this_text.text_content.lower(), keyword_list, use_gpt=False)

        this_text.label = text_label

        if this_text.label != "others":
            elements.append(this_text)

    time_cost_ts = time.clock() - text_selection_time
    print("time cost for text selection: %2.2f s" % time_cost_ts)
    # ts_time_cost_all.append(time_cost_ts)

    # ---------- end -------------------------------

    full_size_org, full_size_grey = pre.read_img(input_img)
    ratio = full_size_org.shape[0]/org.shape[0]

    show = False
    wait_key = 0

    reassign_ids(elements)
    board = merge.show_elements(full_size_org, elements, ratio, show=show, win_name='elements after merging', wait_key=wait_key, line=3)
    board_one_element = merge.show_one_element(full_size_org, elements, ratio, show=show, win_name='elements after merging', wait_key=wait_key, line=3)

    classification_root = pjoin(output_root, 'classification')

    # save all merged elements, clips and blank background
    name = input_img.replace('\\', '/').split('/')[-1][:-4]
    components = merge.save_elements(pjoin(classification_root, name + '.json'), elements, full_size_org.shape, ratio)
    cv2.imwrite(pjoin(classification_root, name + '.jpg'), board)

    print("len(board_one_element): ", len(board_one_element))

    for i in range(len(elements)):
        e_name = str(int(elements[i].id) + 1)
        cv2.imwrite(pjoin(classification_root + '/GUI', name + '-' + e_name + '.jpg'), board_one_element[i])

    print('[Classification Completed] Input: %s Output: %s' % (input_img, pjoin(classification_root, name + '.jpg')))

    # ---------- matching result -----------

    index = input_img.split('/')[-1][:-4]
    app_id = str(index).split('-')[0]

    index_path = pjoin(segment_root, app_id, 'classified_sentences/keyword_index.txt')
    dict_index = {}
    if exists(index_path):
        with open(index_path, 'r') as g:
            for line in g:
                key, value = line.strip().split(':', 1)
                dict_index[key] = value

    for item in elements:
        complete_path = pjoin(segment_root, app_id, 'classified_sentences', item.label + '.txt')
        print("complete_path: ", complete_path)

        if exists(complete_path):

            with open(complete_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Replace line breaks with spaces and strip any extra whitespace
            this_text = ' '.join(content.splitlines()).strip()

            lines = content.splitlines()
            non_empty_lines = [line for line in lines if line.strip() != ""]
            for i in range(len(non_empty_lines)):
                if non_empty_lines[i][0].isalpha():
                    non_empty_lines[i] = non_empty_lines[i][0].upper() + non_empty_lines[i][1:]

            output_data = output_data.append({'screenshot': 's' + str(index), 'id': item.id + 1, 'label': item.label, 'index': dict_index[item.label], 'text': this_text, 'sentences': non_empty_lines}, ignore_index=True)

        else:
            output_data = output_data.append({'screenshot': 's' + str(index), 'id': item.id + 1, 'label': item.label, 'index': "None", 'text': "No information!", 'sentences': "None"},
                                             ignore_index=True)
    return time_cost_ic, time_cost_ts, output_data, board
