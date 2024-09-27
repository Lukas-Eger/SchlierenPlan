import cv2 as cv
import json
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image, ImageEnhance
from preprocessing import extractCameraParameter 
from preprocessing import preprocessing  
from segmentation import segmentation
from config import readParameterFromFile, writeParameterBackToFile, initLocalParameter

%matplotlib qt

#filename for imgSample, imgBack, parameter

plt.close("all")

#1. reading in images
    #img of sample
img = cv.imread('../img/V2/ancycamsy_5mm_.bmp',cv.IMREAD_GRAYSCALE)
    #img of background
imgBack = cv.imread('../img/V2/bildohne5mm.bmp',cv.IMREAD_GRAYSCALE)

#2. loading parameter for preprocessing and segmentation
fileLocal = input("Do you want to load the parameters from selected file?\n(Otherwise the local parameters will be used) [y/n]")
if fileLocal == 'y':
    readParameterFromFile('std_parameter.json') 
    print("Using parameters from selected file!")
elif fileLocal == 'n':
    initLocalParameter()
    print("Using local parameters!")
else:
    print("Wrong input!")
    
#3. loading camera matrix
# calib_df = pd.read_json('calibration.json')
# mtx,dist,new_mtx = extractCameraParameter(calib_df)
# img_undist=cv.undistort(img, mtx, dist, None, new_mtx)
# imgBack_undist=cv.undistort(imgBack, mtx, dist, None, new_mtx)
# img=img_undist.copy()

#3. preprocessing
imgUndist,imgBackUndist,imgCorrectedShading,imgCroppedImage,imgContrast,imgEdgeFiltered,imgEdgeFiltered,imgDeblured = \
preprocessing(img,imgBack)

#5. segmentation
imgSegmented = \
segmentation(imgDeblured)

%matplotlib inline
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

if(response=='y'):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    postfix="_processed_"+date_time
    cv.imwrite((filename+postfix+'.png'), img)
    writeParameterBackToFile(filename+'config'+postfix+'.json')
    