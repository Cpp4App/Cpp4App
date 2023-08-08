import cv2
from os.path import join as pjoin
import time
import json
import numpy as np
from torchvision import models
from torch import nn
import torch
from PIL import Image
import matplotlib.pyplot as plt

import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.file_utils as file
import detect_compo.lib_ip.Component as Compo
from config.CONFIG_UIED import Config
C = Config()


def nesting_inspection(org, grey, compos, ffl_block):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:
            replace = False
            clip_grey = compo.compo_clipping(grey)
            n_compos = det.nested_components_detection(clip_grey, org, grad_thresh=ffl_block, show=False)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def compo_detection(input_img_path, output_root, uied_params,
                    resize_by_height=800, classifier=None, show=False, wai_key=0):

    start = time.clock()
    name = input_img_path.split('/')[-1][:-4] if '/' in input_img_path else input_img_path.split('\\')[-1][:-4]
    ip_root = file.build_directory(pjoin(output_root, "ip"))

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary = pre.binarization(org, grad_min=int(uied_params['min-grad']))

    full_size_org, full_size_grey = pre.read_img(input_img_path)
    ratio = full_size_org.shape[0] / org.shape[0]

    # *** Step 2 *** element detection
    det.rm_line(binary, show=show, wait_key=wai_key)
    uicompos = det.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']))

    # *** Step 3 *** results refinement
    uicompos = det.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=binary.shape)
    uicompos = det.merge_intersected_compos(uicompos)
    det.compo_block_recognition(binary, uicompos)
    if uied_params['merge-contained-ele']:
        uicompos = det.rm_contained_compos_not_in_block(uicompos)
    Compo.compos_update(uicompos, org.shape)
    Compo.compos_containment(uicompos)

    # *** Step 4 ** nesting inspection: check if big compos have nesting element
    uicompos += nesting_inspection(org, grey, uicompos, ffl_block=uied_params['ffl-block'])
    Compo.compos_update(uicompos, org.shape)
    draw.draw_bounding_box(full_size_org, ratio, uicompos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '.jpg'), wait_key=wai_key)

    # # classify icons
    # model = models.resnet18().to('cpu')
    # in_feature_num = model.fc.in_features
    # model.fc = nn.Linear(in_feature_num, 99)
    # # model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding=(3,3), stride=(2,2), bias=False)
    # model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=(3, 3), stride=(2, 2),
    #                         bias=False)
    # # PATH = "C:/ANU/2022 s2/honours project/code/UIED-master/model/model-99-resnet18.pkl"
    # PATH = "./model/model-99-resnet18.pkl"
    # # trained_model = model()
    # model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    #
    # model.eval()
    #
    # # ----------------- try on semantics dataset---------------------
    #
    # # sample_data = np.load('C:/ANU/2022 s2/honours project/code/semantic-icon-classifier-master/data/training_x.npy')
    # #
    # # array = np.reshape(sample_data[0, :, :, :], [32, 32])
    # #
    # # print("array: ", array)
    # #
    # # cv2.imshow("array", array)
    # # cv2.waitKey(0)
    # #
    # # array = array.astype('float32')
    # # array = array / 255
    # # array = (array - array.mean()) / array.std()
    # #
    # # print("array mean: ", array.mean())
    # # print("array std: ", array.std())
    # #
    # # array = array.reshape(1, 1, 32, 32)
    # #
    # # array = torch.tensor(array)
    # # print("array_tensor: ", array)
    # # array_pred_label = model(array)
    # # print("output: ", array_pred_label)
    #
    # # ----------------- end trying ---------------------
    #
    # grey = grey.astype('float32')
    # grey = grey / 255
    # # grey = grey / np.linalg.norm(grey)
    #
    # grey = (grey-grey.mean())/grey.std()
    # print("grey mean: ", grey.mean())
    # print("grey std: ", grey.std())
    #
    # # grey = grey.to(torch.float32)
    #
    # # plt.imshow(Image.fromarray(binary))
    # # plt.show()
    # # cv2.imshow("grey", grey)
    #
    # privacy_compos = []
    # for comp in uicompos:
    #
    #     # cv2.imshow("comp", grey[comp.bbox.row_min:comp.bbox.row_max, comp.bbox.col_min:comp.bbox.col_max])
    #     # cv2.waitKey(0)
    #
    #     # col_mid = int((comp.bbox.col_min+comp.bbox.col_max)/2)
    #     # row_mid = int((comp.bbox.row_min+comp.bbox.row_max)/2)
    #     # comp_crop = grey[max(0, row_mid-16):min(grey.shape[1], row_mid+16), max(0, col_mid-16):min(grey.shape[0], col_mid+16)]
    #     #
    #     # if comp_crop.shape[0] != 32 or comp_crop.shape[1] != 32:
    #     #     print("A component is not classified, size: ", comp_crop.shape)
    #     #     print("col_mid: ", col_mid)
    #     #     print("row_mid: ", row_mid)
    #     #     print("shape[0]: ", comp_crop.shape[0])
    #     #     print("shape[1]: ", comp_crop.shape[1])
    #     #     print("max(0, row_mid-16) and min(binary.shape[1], row_mid+16): ", max(0, row_mid-16), min(grey.shape[1], row_mid+16))
    #
    #     comp_grey = grey[comp.bbox.row_min:comp.bbox.row_max, comp.bbox.col_min:comp.bbox.col_max]
    #
    #     # cv2.imshow("comp_grey", comp_grey)
    #     # cv2.waitKey(0)
    #
    #     # print("comp_crop: ", comp_crop)
    #     # comp_crop = comp_grey.reshape(1, 1, 32, 32)
    #     comp_crop = cv2.resize(comp_grey, (32, 32))
    #     print("comp_crop: ", comp_crop)
    #
    #     # cv2.imshow("comp_crop", comp_crop)
    #     # cv2.waitKey(0)
    #
    #     comp_crop = comp_crop.reshape(1, 1, 32, 32)
    #
    #     comp_tensor = torch.tensor(comp_crop)
    #     comp_tensor = comp_tensor.permute(0, 1, 3, 2)
    #     print("comp_tensor: ", comp_tensor)
    #     # comp_float = comp_tensor.to(torch.float32)
    #     # print("comp_float: ", comp_float)
    #     # pred_label = model(comp_float)
    #     pred_label = model(comp_tensor)
    #     print("output: ", pred_label)
    #     print("label: ", np.argmax(pred_label.cpu().data.numpy(), axis=1))
    #     if np.argmax(pred_label.cpu().data.numpy(), axis=1) in [72.0, 42.0, 77.0, 91.0, 6.0, 89.0, 40.0, 43.0, 82.0, 3.0, 68.0,
    #                                                             49.0, 56.0, 89.0]:
    #         privacy_compos.append(comp)
    #
    # draw.draw_bounding_box(org, privacy_compos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '.jpg'), wait_key=wai_key)

    # *** Step 5 *** image inspection: recognize image -> remove noise in image -> binarize with larger threshold and reverse -> rectangular compo detection
    # if classifier is not None:
    #     classifier['Image'].predict(seg.clipping(org, uicompos), uicompos)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     uicompos = det.rm_noise_in_large_img(uicompos, org)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     det.detect_compos_in_img(uicompos, binary_org, org)
    #     draw.draw_bounding_box(org, uicompos, show=show)
    # if classifier is not None:
    #     classifier['Noise'].predict(seg.clipping(org, uicompos), uicompos)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     uicompos = det.rm_noise_compos(uicompos)

    # *** Step 6 *** element classification: all category classification
    # if classifier is not None:
    #     classifier['Elements'].predict([compo.compo_clipping(org) for compo in uicompos], uicompos)
    #     draw.draw_bounding_box_class(org, uicompos, show=show, name='cls', write_path=pjoin(ip_root, 'result.jpg'))
    #     draw.draw_bounding_box_class(org, uicompos, write_path=pjoin(output_root, 'result.jpg'))

    # *** Step 7 *** save detection result

    Compo.compos_update(uicompos, org.shape)
    file.save_corners_json(pjoin(ip_root, name + '.json'), uicompos)
    # file.save_corners_json(pjoin(ip_root, name + '.json'), uicompos, full_size_org, ratio)

    cd_time = time.clock() - start
    print("[Compo Detection Completed in %.3f s] Input: %s Output: %s" % (cd_time, input_img_path, pjoin(ip_root, name + '.json')))
    return cd_time
