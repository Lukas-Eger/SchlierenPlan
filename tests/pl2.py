"""
Pipline for Digital Image Processing

pl2: Pipline 2

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

imgFolder = '../img/V0/'
imgTestSubject = 'v_smart_10'
imgInputSuffix = '_raw'
imgOutputSuffix = '_pl2'
bgImgSuffix = '_bg'
imgDataType = '.jpg'


#set parameter
parameter = {
    "bgImgAvailable" : False,
    
    "doShadingCorrection": False,
    
    "cropImage": False,
    
    "setupContrast": True,
    "alpha": 2.2,   #lower contrast: alpha < 1, higher contrast alpha > 1 
    "beta": -60,    #brightness -127 < beta < 127
    
    "edgeEnhancement": False,
    "ddepth": cv.CV_8U,    #CV_8U, CV_16S, CV_32F, CV64_F
    "kernelSizeLaplacian": 7,
    
    "deblureImage": True,
    "meanFilteredImage": True,
    "kernelSizeMeanFilter": 30,
    "gaussianFilteredImage": False,
    "kernelSizeGaussianFilter": 31,
    "standardDeviation": 0,     #calculated from kernelSize
    "medianFilteredImage": False,
    "kernelSizeMedianFilter": 31,
    "bilateralFilteredImage": False,
    "bilateralFilterDiameter": 31,   #diameter of each pixel neighborhood that is used during filtering
    "bilateralFilterSigma": 250  #<10: no effect >150: huge effect on outcome
}

#read images as grayscale images
img = cv.imread(imgFolder+imgTestSubject+imgInputSuffix+imgDataType, cv.IMREAD_GRAYSCALE)
imgInput = img.copy()
if parameter["bgImgAvailable"]:
    bgImg = cv.imread(imgFolder+imgTestSubject+bgImgSuffix+imgDataType, cv.IMREAD_GRAYSCALE)


#pre-processing pipline
#   1. shading correction
if parameter["bgImgAvailable"] and parameter["doShadingCorrection"]:
#   img = img.mean() * img./bgImg
    imgCorrectedShading = img.copy()

#   2. crop image
if parameter["cropImage"]:
    #implementation with help of PIL module
    imgCroppedImage = img.copy()
    
#   3. contrast image
if parameter["setupContrast"]:
    img = cv.convertScaleAbs(img, alpha=parameter["alpha"], beta=parameter["beta"])
    imgContrast = img.copy()

#   x. edge enhancement with different approaches (perhaps after debluring image)
if parameter["edgeEnhancement"]:
    img = cv.Laplacian(img, parameter["ddepth"], ksize=parameter["kernelSizeLaplacian"])
    cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    imgLaplacianFiltered = img.copy()

#   x. deblure img with different approaches (perhaps before edge enhancement)
if parameter["deblureImage"]:
#       1 mean filter:
        imgMeanFiltered = cv.blur(img,(parameter["kernelSizeMeanFilter"],parameter["kernelSizeMeanFilter"]))
#       2 gaussian filter
        imgGaussianFiltered = cv.GaussianBlur(img, (parameter["kernelSizeGaussianFilter"],parameter["kernelSizeGaussianFilter"]), parameter["standardDeviation"])
#       3 median filter
        imgMedianFiltered = cv.medianBlur(img, ksize=parameter["kernelSizeMedianFilter"])
#       4 Bilateral Filtering (keeping edges sharp)
        imgBilateralFiltered = cv.bilateralFilter(img, parameter["bilateralFilterDiameter"], parameter["bilateralFilterSigma"], parameter["bilateralFilterSigma"])
        if parameter["meanFilteredImage"]:
            img = imgMeanFiltered.copy()
        elif parameter["gaussianFilteredImage"]:
            img = imgGaussianFiltered.copy()
        elif parameter["medianFilteredImage"]:
            img = imgMedianFiltered.copy()
        elif parameter["bilateralFilteredImage"]:
            img = imgBilateralFiltered.copy()
        else:
            print("Warning: No debluring Filter was chosen although active\n")
    
#plotting of result (and intermediary results) of pre-processing pipline
#   0. plotting input image
plt.figure()
fig0 = plt.subplot()
fig0.imshow(imgInput, cmap = 'gray')
plt.title("0. Input Image")
plt.axis('off')

#   1. plotting shading corrected image
plt.figure()
fig1 = plt.subplot()
if parameter["bgImgAvailable"] and parameter["doShadingCorrection"]:
    fig1.imshow(imgCorrectedShading, cmap = 'gray')    #show only cutout area
plt.title("1. Shading Correction")
plt.axis('off')

#   2. plotting cropped image
plt.figure()
fig2 = plt.subplot()
if parameter["cropImage"]:
    fig2.imshow(imgCroppedImage, cmap = 'gray')
plt.title("2. Cropped Image")
plt.axis('off')

#   3. plotting contrasted image
plt.figure()
fig3 = plt.subplot()
if parameter["setupContrast"]:
    fig3.imshow(imgContrast, cmap = 'gray')
plt.title("3. Setup Contrast")
plt.axis('off')

#   4. plotting edge enhanced image
plt.figure()
fig4 = plt.subplot()
if parameter["edgeEnhancement"]:
    fig4.imshow(imgLaplacianFiltered, cmap = 'gray')
plt.title("4. Edge Enhancement (Laplacian Filtered)")
plt.axis('off')

#   5. plotting deblured image
if parameter["deblureImage"]:
    plt.figure()
    fig5 = plt.subplot()
    fig5.imshow(imgMeanFiltered, cmap = 'gray')
    plt.title('5. Deblured Images\nMean Filtered')
    plt.axis('off')
    
    plt.figure()
    fig6 = plt.subplot()
    fig6.imshow(imgGaussianFiltered, cmap = 'gray')
    plt.title('5. Deblured Images\nGaussian Filtered')
    plt.axis('off')
    
    plt.figure()
    fig7 = plt.subplot()
    fig7.imshow(imgMedianFiltered, cmap = 'gray')
    plt.title('5. Deblured Images\nMedian Filtered')
    plt.axis('off')
    
    plt.figure()
    fig8 = plt.subplot()
    fig8.imshow(imgBilateralFiltered, cmap = 'gray')
    plt.title('5. Deblured Images\nBilateral Filtered')
    plt.axis('off')

#edge enhancement doesn't work properly
#cutout
#detect refractive index via cv.canny???