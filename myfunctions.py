# CV Assignment 2 - functions used frequently
# Kzesniak Magdalena-Izabela 161044
# cs161044@uniwa.gr

# -------------------------
# needed imports
# ---------------------------

import cv2
import re
from numpy import random as rn
import numpy as np


# resizing the given image to x % of original size
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimensions = (width, height)  # new dimension of images
    # resize image
    resized_image = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
    return resized_image


# saving images to OutputFiles directory and showing them
def show_image(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
    cv2.imshow(title, img)
    filename = "OutputFiles/" + re.sub('\s+', '', title).lower() + ".jpg"
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imwrite(filename, img)  # saving image in directory OutputFiles
    cv2.waitKey()


# noise_on_image is used to add noise on image( amount depends on the percentage) using this formula and the function
# rand() :   eij(new) =  eij + noise*rand()* eij - noise*rand()* eij
# works both for rgb and greyscale
def noise_on_image(img, n_percentage):
    temp = img + n_percentage * img * rn.random(img.shape) - n_percentage * img * rn.random(img.shape)
    return temp.astype(np.uint8)  # image has to be uint8
