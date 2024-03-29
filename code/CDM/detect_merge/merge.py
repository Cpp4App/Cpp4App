import json
import cv2
import numpy as np
from os.path import join as pjoin
import os
import time
import shutil

from detect_merge.Element import Element
from torchvision import models
from torch import nn
import torch

import detect_compo.lib_ip.ip_preprocessing as pre

# ----------------- load pre-trained classification model ----------------

# model = models.resnet18().to('cpu')
# in_feature_num = model.fc.in_features
# model.fc = nn.Linear(in_feature_num, 99)
# model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=(3, 3), stride=(2, 2),
#                         bias=False)
#
# PATH = "./model/model-99-resnet18.pkl"
# model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
#
# model.eval()

# ----------------- end loading ------------------------------------------

# information_type = {'Name':['name', 'first name', 'last name', 'full name', 'real name', 'surname', 'family name', 'given name'],
#                         'Birthday':['birthday', 'date of birth', 'birth date', 'DOB', 'dob full birthday'],
#                         'Address':['address', 'mailing address', 'physical address', 'postal address', 'billing address', 'shipping address'],
#                         'Phone':['phone', 'phone number', 'mobile', 'mobile phone', 'mobile number', 'telephone', 'telephone number', 'call'],
#                         'Email':['email', 'e-mail', 'email address', 'e-mail address'],
#                         'Contacts':['contacts', 'phone-book', 'phone book'],
#                         'Location':['location', 'locate', 'place', 'geography', 'geo', 'geo-location', 'precision location'],
#                         'Camera':['camera', 'photo', 'scan', 'album', 'picture', 'gallery', 'photo library', 'storage', 'image', 'video'],
#                         'Microphone':['microphone', 'voice, mic', 'speech', 'talk'],
#                         'Financial':['credit card', 'pay', 'payment', 'debit card', 'mastercard', 'wallet'],
#                         'IP':['IP', 'Internet Protocol', 'IP address', 'internet protocol address'],
#                         'Cookies':['cookies', 'cookie'],
#                         'Social':['facebook', 'twitter']}

def show_elements(org_img, eles, ratio, show=False, win_name='element', wait_key=0, shown_resize=None, line=2):
    color_map = {'Text':(0, 0, 255), 'Compo':(0, 255, 0), 'Block':(0, 255, 0), 'Text Content':(255, 0, 255)}
    img = org_img.copy()
    for ele in eles:
        color = color_map[ele.category]
        ele.visualize_element(img=img, color=color, line=line, ratio=ratio)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        cv2.imshow(win_name, img_resize)
        cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(win_name)
    return img_resize

def show_one_element(org_img, eles, ratio, show=False, win_name='element', wait_key=0, shown_resize=None, line=2):
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 0), 'Text Content': (255, 0, 255)}
    all_img = []
    for ele in eles:
        img = org_img.copy()
        color = color_map[ele.category]
        ele.visualize_element(img=img, color=color, line=line, ratio=ratio)
        img_resize = img
        all_img.append(img_resize)
        if shown_resize is not None:
            img_resize = cv2.resize(img, shown_resize)
        if show:
            cv2.imshow(win_name, img_resize)
            cv2.waitKey(wait_key)
            if wait_key == 0:
                cv2.destroyWindow(win_name)
    return all_img


def save_elements(output_file, elements, img_shape, ratio=1):
    components = {'compos': [], 'img_shape': img_shape}
    for i, ele in enumerate(elements):

        if ratio != 1:
            ele.resize(ratio)
            ele.width = ele.col_max - ele.col_min
            ele.height = ele.row_max - ele.row_min

        c = ele.wrap_info()
        # c['id'] = i
        components['compos'].append(c)
    json.dump(components, open(output_file, 'w'), indent=4)
    return components


def reassign_ids(elements):
    for i, element in enumerate(elements):
        element.id = i


def refine_texts(texts, img_shape):
    refined_texts = []
    # for text in texts:
    #     # remove potential noise
    #     if len(text.text_content) > 1 and text.height / img_shape[0] < 0.075:
    #         refined_texts.append(text)

    for text in texts:
        # remove potential noise
        if text.height / img_shape[0] < 0.075:
            refined_texts.append(text)

    return refined_texts


def merge_text_line_to_paragraph(elements, max_line_gap=5):
    texts = []
    non_texts = []
    for ele in elements:
        if ele.category == 'Text':
            texts.append(ele)
        else:
            non_texts.append(ele)

    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                inter_area, _, _, _ = text_a.calc_intersection_area(text_b, bias=(0, max_line_gap))
                if inter_area > 0:
                    text_b.element_merge(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return non_texts + texts


def refine_elements(compos, texts, input_img_path, intersection_bias=(2, 2), containment_ratio=0.8, ):
    '''
    1. remove compos contained in text
    2. remove compos containing text area that's too large
    3. store text in a compo if it's contained by the compo as the compo's text child element
    '''

    # resize_by_height = 800
    # org, grey = pre.read_img(input_img_path, resize_by_height)
    #
    # grey = grey.astype('float32')
    # grey = grey / 255
    #
    # grey = (grey - grey.mean()) / grey.std()

    elements = []
    contained_texts = []

    # classification_start_time = time.time()

    for compo in compos:
        is_valid = True
        text_area = 0
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(text, bias=intersection_bias)
            if inter > 0:
                # the non-text is contained in the text compo
                if ioa >= containment_ratio:
                    is_valid = False
                    break
                text_area += inter
                # the text is contained in the non-text compo
                if iob >= containment_ratio and compo.category != 'Block':
                    contained_texts.append(text)
            # print("id: ", compo.id)
            # print("text.text_content: ", text.text_content)
            # print("is_valid: ", is_valid)
            # print("inter: ", inter)
            # print("iou: ", iou)
            # print("ioa: ", ioa)
            # print("iob: ", iob)
            # print("text_area: ", text_area)
            # print("compo.area: ", compo.area)
        if is_valid and text_area / compo.area < containment_ratio:
            # for t in contained_texts:
            #     t.parent_id = compo.id
            # compo.children += contained_texts

            # --------- classification ----------

            # comp_grey = grey[compo.row_min:compo.row_max, compo.col_min:compo.col_max]
            #
            # comp_crop = cv2.resize(comp_grey, (32, 32))
            #
            # comp_crop = comp_crop.reshape(1, 1, 32, 32)
            #
            # comp_tensor = torch.tensor(comp_crop)
            # comp_tensor = comp_tensor.permute(0, 1, 3, 2)
            #
            # pred_label = model(comp_tensor)
            #
            # if np.argmax(pred_label.cpu().data.numpy(), axis=1) in [72.0, 42.0, 77.0, 91.0, 6.0, 89.0, 40.0, 43.0, 82.0,
            #                                                         3.0, 68.0, 49.0, 56.0, 89.0]:
            #     elements.append(compo)

            # --------- end classification ----------

            elements.append(compo)
    # time_cost_ic = time.time() - classification_start_time
    # print("time cost for icon classification: %2.2f s" % time_cost_ic)

    # text_selection_time = time.time()

    # elements += texts
    for text in texts:
        if text not in contained_texts:
            elements.append(text)

            # ---------- Simulate keyword search -----------

            # for key in keyword_list:
            #     for w in keyword_list[key]:
            #         if w in text.text_content.lower():
            #             elements.append(text)

            # ---------- end -------------------------------

    # time_cost_ts = time.time() - text_selection_time
    # print("time cost for text selection: %2.2f s" % time_cost_ts)

    # return elements, time_cost_ic, time_cost_ts
    return elements


def check_containment(elements):
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1:
                elements[j].children.append(elements[i])
                elements[i].parent_id = elements[j].id
            if relation == 1:
                elements[i].children.append(elements[j])
                elements[j].parent_id = elements[i].id


def remove_top_bar(elements, img_height):
    new_elements = []
    max_height = img_height * 0.04
    for ele in elements:
        if ele.row_min < 10 and ele.height < max_height:
            continue
        new_elements.append(ele)
    return new_elements


def remove_bottom_bar(elements, img_height):
    new_elements = []
    for ele in elements:
        # parameters for 800-height GUI
        if ele.row_min > 750 and 20 <= ele.height <= 30 and 20 <= ele.width <= 30:
            continue
        new_elements.append(ele)
    return new_elements


def compos_clip_and_fill(clip_root, org, compos):
    def most_pix_around(pad=6, offset=2):
        '''
        determine the filled background color according to the most surrounding pixel
        '''
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1
        most = []
        for i in range(3):
            val = np.concatenate((org[up:row_min - offset, left:right, i].flatten(),
                            org[row_max + offset:bottom, left:right, i].flatten(),
                            org[up:bottom, left:col_min - offset, i].flatten(),
                            org[up:bottom, col_max + offset:right, i].flatten()))
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)

    bkg = org.copy()
    cls_dirs = []
    for compo in compos:
        cls = compo['class']
        if cls == 'Background':
            compo['path'] = pjoin(clip_root, 'bkg.png')
            continue
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo['id']) + '.jpg')
        compo['path'] = c_path
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)

        position = compo['position']
        col_min, row_min, col_max, row_max = position['column_min'], position['row_min'], position['column_max'], position['row_max']
        cv2.imwrite(c_path, org[row_min:row_max, col_min:col_max])
        # Fill up the background area
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), most_pix_around(), -1)
    cv2.imwrite(pjoin(clip_root, 'bkg.png'), bkg)


def merge(img_path, compo_path, text_path, merge_root=None, is_paragraph=False, is_remove_top_bar=False, is_remove_bottom_bar=False, show=False, wait_key=0):
    compo_json = json.load(open(compo_path, 'r'))
    text_json = json.load(open(text_path, 'r'))

    # load text and non-text compo
    ele_id = 0
    compos = []
    for compo in compo_json['compos']:
        element = Element(ele_id, (compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']), compo['class'])
        compos.append(element)
        ele_id += 1
    texts = []
    for text in text_json['texts']:
        element = Element(ele_id, (text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text', text_content=text['content'])
        texts.append(element)
        ele_id += 1
    if compo_json['img_shape'] != text_json['img_shape']:
        resize_ratio = compo_json['img_shape'][0] / text_json['img_shape'][0]
        for text in texts:
            text.resize(resize_ratio)

    # check the original detected elements
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (compo_json['img_shape'][1], compo_json['img_shape'][0]))
    ratio = img.shape[0] / img_resize.shape[0]

    show_elements(img, texts + compos, ratio, show=show, win_name='all elements before merging', wait_key=wait_key, line=3)

    # refine elements
    texts = refine_texts(texts, compo_json['img_shape'])
    elements = refine_elements(compos, texts, img_path)
    if is_remove_top_bar:
        elements = remove_top_bar(elements, img_height=compo_json['img_shape'][0])
    if is_remove_bottom_bar:
        elements = remove_bottom_bar(elements, img_height=compo_json['img_shape'][0])
    if is_paragraph:
        elements = merge_text_line_to_paragraph(elements, max_line_gap=7)
    reassign_ids(elements)
    check_containment(elements)
    board = show_elements(img, elements, ratio, show=show, win_name='elements after merging', wait_key=wait_key, line=3)

    # save all merged elements, clips and blank background
    name = img_path.replace('\\', '/').split('/')[-1][:-4]
    components = save_elements(pjoin(merge_root, name + '.json'), elements, img_resize.shape)
    cv2.imwrite(pjoin(merge_root, name + '.jpg'), board)
    print('[Merge Completed] Input: %s Output: %s' % (img_path, pjoin(merge_root, name + '.jpg')))
    return board, components
    # return this_ic_time, this_ts_time
