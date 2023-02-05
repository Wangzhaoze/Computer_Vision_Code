import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import cv2
from math import floor, ceil
from .cv_utils import *


#####################################################################################
############################# images data loading ###################################
#####################################################################################

IMAGE_DIR = './dataset/FAU_Panorama/'
images = ["1.jpg",  "2.jpg",  "3.jpg",  "4.jpg", "5.jpg",
          "6.jpg", "7.jpg", "8.jpg", "9.jpg"]

image_data_list = []

for i, image_name in enumerate(images):
    image_data_dict = {}
    image_data_dict['file'] = image_name

    image_path = os.path.join(IMAGE_DIR, image_name)
    image_data_dict['img'] = cv2.imread(image_path)
    image_data_dict['img'] = cv2.resize(image_data_dict['img'], None, fx=0.5, fy=0.5)

    image_data_dict['id'] = i + 1

    image_data_dict['HtoReference'] = np.eye(3)
    image_data_dict['HtoPrev'] = np.eye(3)
    image_data_dict['HtoNext'] = np.eye(3)

    assert len(image_data_dict['img']) > 0

    image_data_list.append(image_data_dict)

    print('successfully loaded image: {id}    image name: {name}    image size: {size}  '.format(id=image_data_dict['id'],
                                                                                                 name=image_data_dict['file'],
                                                                                                 size=image_data_dict['img'].shape))
print('images loaded successfully!')


#####################################################################################
################################# stitching #########################################
#####################################################################################


# feature detection
image_data_dicts = feature_detection(image_data_list)

for i in range(1, len(image_data_dicts)):

    # feature matching
    matches = computeMatches(image_data_dicts[i - 1], image_data_dicts[i])


    # create match image
    matchImg = createMatchImage(image_data_dicts[i - 1], image_data_dicts[i], matches)
    h = 200
    w = int((float(matchImg.shape[1]) / matchImg.shape[0]) * h)
    matchImg = cv2.resize(matchImg, (w, h))
    name = "Matches (" + str(i - 1) + "," + str(i) + ") " + image_data_dicts[i - 1]['file'] + " - " + \
           image_data_dicts[i]['file']
    cv2.namedWindow(name)
    cv2.moveWindow(name, int(10 + w * ((i - 1) % 2)), int(10 + (h + 30) * ((i - 1) / 2)))
    cv2.imshow(name, matchImg)

    # compute homograpy with RANSAC
    H = computeHomographyRansac(image_data_dicts[i - 1], image_data_dicts[i], matches, 1000, 2.0)
    image_data_dicts[i]['HtoPrev'] = np.linalg.inv(H)
    image_data_dicts[i - 1]['HtoNext'] = H


# create stitch image
simg = createStichedImage(image_data_dicts)
cv2.imwrite("output.png", simg)