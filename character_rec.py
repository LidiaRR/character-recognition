import pandas as pd
import numpy as np

from glob import glob

import cv2 as cv

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

## Function definition

# Calculates the descriptors for an image
#   - List of descriptors:
#       1. First 20 Fourier descriptors of the main contour of the inverse of the image, both real and imaginary parts separately
#       2. First 20 Fourier descriptors of the main contour of the eroded version of the inverse of the image, both real and imaginary parts separately
#       3. Number of countours that the inverse of the image has
#   - Input:
#       1. img: an image represented as a matrix, either in color or black & white, containing an uppercase letter written in a dark color over a light color background
#   - Output: a vector with the vectors described above for the given image
def get_descriptors(img):
    img2gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(img2gray, 127, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((35,35),np.uint8)
    erosion = cv.erode(mask, kernel,iterations = 1)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    main_cont = max(contours, key=cv.contourArea)
    contours2, _ = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    main_cont2 = max(contours2, key=cv.contourArea)

    contour_points = main_cont[:, 0, :]
    contour_complex = np.empty(contour_points.shape[0], dtype=complex)
    contour_points2 = main_cont2[:, 0, :]
    contour_complex2 = np.empty(contour_points2.shape[0], dtype=complex)

    for i in range(contour_points.shape[0]):
        contour_complex[i] = complex(contour_points[i, 0], contour_points[i, 1])
    for i in range(contour_points2.shape[0]):
        contour_complex2[i] = complex(contour_points2[i, 0], contour_points2[i, 1])

    fourier_result = np.fft.fft(contour_complex)
    fourier_result2 = np.fft.fft(contour_complex2)

    n_descriptors = 20

    result = []
    for i in range(n_descriptors):
        result += [fourier_result[i].real, fourier_result[i].imag]
    for i in range(n_descriptors):
        result += [fourier_result2[i].real, fourier_result2[i].imag]
    result += [len(contours2)]

    return result


## Main code

image_set = glob('./archive/Img/*.png')
descriptors = []
labels = []
for el in image_set:
    labels += [el[17:20]]
    descriptors += [get_descriptors(cv.imread(el))]

X_train, X_test, Y_train, Y_test = train_test_split(descriptors, labels, test_size=0.2)
classifier = MLPClassifier(hidden_layer_sizes=(200,))
classifier.fit(X_train, Y_train) 

accuracy = classifier.score(X_test, Y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

