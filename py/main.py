import cv2 as cv
import json
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
from preprocessing import extractCameraParameter 
from preprocessing import preprocessing  
from segmentation import segmentation

%matplotlib qt

filename="test"

#reading in images
#   img of sample
img=cv.imread('../img/V2/ancycamsy_5mm_.bmp',cv.IMREAD_GRAYSCALE)
#   img of background
imgBack=cv.imread('../img/V2/bildohne5mm.bmp',cv.IMREAD_GRAYSCALE)

#einlesne der Paremter
localuse= input("möchten sie die lokalen Parameter oder die aus einer Datei verwenden(l/d)")
if(localuse=="l"):

    parameter = {
        "bgImgAvailable" : True,
        
        "doShadingCorrection": True,
        
        "cropImage": True,
        "x_start":0,
        "y_start":0,
        "x_end":0,
        "y_end":0,
       
        
        
        "setupContrast": True,
        "alpha": 1.8,   #lower contrast: alpha < 1, higher contrast alpha > 1 
        "beta": -30,    #brightness -127 < beta < 127
        
        "edgeEnhancement": True,
        "laplacianFilteredImage": False,
        "ddepthLaplacian": cv.CV_8U,    #CV_8U, CV_16S, CV_32F, CV_64F
        "kernelSizeLaplacian": 7,
        "pillowSharpendImage": True,
        "factorSharpness": 21.0,
        "sobeFilteredImage": False,
        "kernelSizeSobelFilter": 7,
        "ddepthSobel": cv.CV_8U,        #CV_8U, CV_16S, CV_32F, CV_64F
        
        "deblureImage": True,
        "meanFilteredImage": False,
        "kernelSizeMeanFilter": 5,
        "gaussianFilteredImage": False,
        "kernelSizeGaussianFilter": 31,
        "standardDeviation": 0,     #calculated from kernelSize
        "medianFilteredImage": True,
        "kernelSizeMedianFilter":9,
        "bilateralFilteredImage": False,
        "bilateralFilterDiameter": 31,   #diameter of each pixel neighborhood that is used during filtering
        "bilateralFilterSigma": 250  #<10: no effect >150: huge effect on outcome
    }
else:
    with open('config.json', 'r') as cf:
      parameter = json.load(cf)

plt.close("all")
#einlesen Kamerametrix
calib_df = pd.read_json('calibration.json')

#undistort
mtx,dist,new_mtx = extractCameraParameter(calib_df)
img_undist = cv.undistort(img, mtx, dist, None, new_mtx)
imgBack_undist = cv.undistort(imgBack, mtx, dist, None, new_mtx)
img= img_undist.copy()
#preprocessing

imgCorrectedShading,imgCroppedImage,imgContrast,imgEdgeFiltered,imgEdgeFiltered,imgDeblured = \
preprocessing(img_undist,imgBack_undist ,parameter)
%matplotlib inline
#segmentation

imgSegmented=segmentation(imgDeblured,parameter)



plt.figure()
fig7 = plt.subplot()
fig7.imshow(imgCroppedImage, cmap = 'gray')
plt.title('x. imgCorrectedShading')
plt.axis('off')
plt.figure()
plt.imshow(imgContrast,  cmap = 'gray')
plt.title("imgContrast scheis")
plt.show()

plt.figure()
plt.imshow(imgSegmented,  cmap = 'gray')
plt.title("imgSegmented scheis")
plt.show()

plt.figure()
plt.imshow(imgDeblured,  cmap = 'gray')
plt.title("geiler scheis")
plt.show()
response = input("möchten sie das result speichern (y/n)")

if(response=="y"):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    postfix="_processed_"+date_time
    cv.imwrite((filename+postfix+'.png'), img)
    with open(filename+'config'+postfix+'.json', 'w') as json_file:
      json.dump(parameter, json_file)