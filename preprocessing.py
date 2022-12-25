import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Pandas configuration
pd.set_option('display.max_columns', None)
print(cv2.__version__)

import pydicom
dicom_root = 'F:/praca-magisterska/DANE/zdrowe/'
patients = [ x for x in os.listdir(dicom_root) ]
print('Patient count: {}'.format(len(patients)))

index = 0
IMG_PX_SIZE = 224


def rescale_correction(s):
    s.image = s.pixel_array * s.RescaleSlope + s.RescaleIntercept


# Load lung images and apply a segmentation algorithm to each
def load_patient(patient_id):
    files = glob.glob(dicom_root + patient_id)
    slices = []
    for f in files:
        dcm = pydicom.read_file(f)
        rescale_correction(dcm)
        slices.append(dcm)

    slices = sorted(slices, key=lambda x: x.SliceLocation)
    return slices


for patient_no in sorted(patients):
    pat = load_patient(patient_no)
    print(patient_no)

    img = pat[index].image.copy()

    # threshold HU > -300
    img[img > -300] = 255
    img[img < -300] = 0
    img = np.uint8(img)

    # find surrounding torso from the threshold and make a mask
    contours, im2 = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, [largest_contour], 255)

    # apply mask to threshold image to remove outside.
    img = ~img
    img[(mask == 0)] = 0

    # apply closing to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    # apply mask to image
    img2 = pat[index].image.copy()
    img2[(img == 0)] = -2000  # <-- Larger than threshold value
    points = np.argwhere(img2 != -2000)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
    x, y, w, h = x - 10, y - 10, w + 20, h + 20  # make the box bigger
    crop = img2[y:y + h, x:x + w]  # create a cropped region of the gray image
    plt.imshow(crop, cmap='gray')
    plt.show()
    resized = cv2.resize(crop, (IMG_PX_SIZE, IMG_PX_SIZE))
    plt.imsave('F:/praca-magisterska/DANE/zdrowe-png/' + patient_no.replace('.dcm', '.png'), resized, cmap='gray')
    # img = cv2.imread('C:/Users/Jerem/Desktop/plucka/chore/' + patient_no.replace('.dcm', '.png'))
