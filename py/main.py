import sys
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
from plotter import plotResults

%matplotlib qt
#matplotlib inline

plt.close("all")

#filename for imgSample, imgBack, parameter

#reading in images
    #img of sample
img = cv.imread('../img/V2/ancycamsy_5mm_.bmp', cv.IMREAD_GRAYSCALE)
    #img of background
imgBack = cv.imread('../img/V2/bildohne5mm.bmp', cv.IMREAD_GRAYSCALE)

#loading parameter for preprocessing and segmentation
fileLocalResponse = input("Do you want to load the parameters from selected file?\n(Otherwise the local parameters will be used) [y/n]: ")
if fileLocalResponse == 'y':
    readParameterFromFile('std_parameter.json') 
    print("Using parameters from selected file!\n")
elif fileLocalResponse == 'n':
    initLocalParameter()
    print("Using local parameters!\n")
else:
    sys.exit("Wrong input!")
    
#I. preprocessing
img, _, _, _, imgCropped, imgContrast, imgEdgeFiltered, imgDeblured = preprocessing(img, imgBack)

#II. segmentation
img, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented = segmentation(img)

#show (intermediate) results
plotResults(img, imgCropped, imgContrast, imgEdgeFiltered, imgDeblured, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented)

#save result and parameters
# saveResponse = input("möchten sie das result speichern (y/n)")
# if(response=='y'):
#     now = datetime.now()
#     date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
#     postfix="_processed_"+date_time
#     #cv.imwrite((filename+postfix+'.png'), img)
#     #writeParameterBackToFile(filename+'config'+postfix+'.json')
#     writeParameterBackToFile('std_parameter.json')
    