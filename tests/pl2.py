"""
Pipline for Digital Image Processing

pl2: Pipline 2

"""

import cv2 as cv
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

imgFolder = '../img/V1/'
imgTestSubject = 'h_std_05'
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
    "alpha": 2.0,   #lower contrast: alpha < 1, higher contrast alpha > 1 
    "beta": -60,    #brightness -127 < beta < 127
    
    "edgeEnhancement": True,
    "laplacianFilteredImage": False,
    "ddepthLaplacian": cv.CV_8U,    #CV_8U, CV_16S, CV_32F, CV_64F
    "kernelSizeLaplacian": 7,
    "pillowSharpendImage": False,
    "factorSharpness": 7.0,
    "sobeFilteredImage": False,
    "kernelSizeSobelFilter": 7,
    "ddepthSobel": cv.CV_8U,        #CV_8U, CV_16S, CV_32F, CV_64F
    
    
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
#       1 laplacian filter
        imgLaplacianFiltered = cv.Laplacian(img, parameter["ddepthLaplacian"], ksize=parameter["kernelSizeLaplacian"])
        cv.normalize(imgLaplacianFiltered, imgLaplacianFiltered, 0, 255, cv.NORM_MINMAX)
#       2 from pillow: image enhancer
        img_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(img_pil)
        imgPillowSharpend = enhancer.enhance(parameter["factorSharpness"])
        imgPillowSharpend = np.array(imgPillowSharpend)
#       3 sobel filter
        imgSobelFiltered = cv.Sobel(img, ddepth=parameter["ddepthSobel"], dx=1, dy=1, ksize=parameter["kernelSizeSobelFilter"])
        cv.normalize(imgSobelFiltered, imgSobelFiltered, 0, 255, cv.NORM_MINMAX)
        if parameter["laplacianFilteredImage"]:
            img = imgLaplacianFiltered.copy()
        elif parameter["pillowSharpendImage"]:
            img = imgPillowSharpend.copy()
        elif parameter["sobeFilteredImage"]:
            img = imgSobelFiltered.copy()
        else:
            print("Warning: No edge enhancement was chosen although active\n")

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
if parameter["edgeEnhancement"]:
    plt.figure()
    fig4 = plt.subplot()
    fig4.imshow(imgLaplacianFiltered, cmap = 'gray')
    plt.title("x. Edge Enhancement\n(Laplacian Filtered)")
    plt.axis('off')
    
    plt.figure()
    fig5 = plt.subplot()
    fig5.imshow(imgPillowSharpend, cmap = 'gray')
    plt.title("x. Edge Enhancement\n(Pillow Sharpend)")
    plt.axis('off')
    
    plt.figure()
    fig5 = plt.subplot()
    fig5.imshow(imgSobelFiltered, cmap = 'gray')
    plt.title("x. Edge Enhancement\n(Sobel Filtered)")
    plt.axis('off')

#   5. plotting deblured image
if parameter["deblureImage"]:
    plt.figure()
    fig7 = plt.subplot()
    fig7.imshow(imgMeanFiltered, cmap = 'gray')
    plt.title('x. Deblured Images\nMean Filtered')
    plt.axis('off')
    
    
    plt.figure()
    fig8 = plt.subplot()
    fig8.imshow(imgGaussianFiltered, cmap = 'gray')
    plt.title('x. Deblured Images\nGaussian Filtered')
    plt.axis('off')
    
    plt.figure()
    fig9 = plt.subplot()
    fig9.imshow(imgMedianFiltered, cmap = 'gray')
    plt.title('x. Deblured Images\nMedian Filtered')
    plt.axis('off')
    
    plt.figure()
    fig10 = plt.subplot()
    fig10.imshow(imgBilateralFiltered, cmap = 'gray')
    plt.title('x. Deblured Images\nBilateral Filtered')
    plt.axis('off')

#debluring -> edge_enhancement ??
#edge enhancement doesn't work properly 
#rectangular cutout without back edges
#detect refractive index via cv.canny???


#save pictures in directory with datetime (but only if desired)
#save also parameters of pictures