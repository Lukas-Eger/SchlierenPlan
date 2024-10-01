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
#%matplotlib inline

plt.close("all")

#   img:                '../img/Tyy/xx_Tyy_img.ext'
#   imgBack:            '../img/Tyy/xx_Tyy_imgBack.ext'
#   imgResult:          '../img/Tyy/xx_Tyy_imgResult_YYYY_mm_dd_HH_MM.ext'
#   parameterResult:    '../img/Tyy/xx_Tyy_parameter_YYYY_mm_dd_HH_MM.json'
#--------- PLEASE CONFIGURE ------------
Tyy  = 'T02'    #number of test experiment yy
xx   = '01'     #number of image xx
YYYY = '2024'   #year YYYY
mm   = '10'     #month mm
dd   = '01'     #day dd
HH   = '21'     #hour HH
MM   = '54'     #minute MM
ext  = '.bmp'   #file extension: .bmp, .jpg, .png
#----------------------------------------
imgDir                 = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_img'+ext
imgBackDir             = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_imgBack'+ext
imgResultDir           = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_imgResult_'
parameterResultDir     = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_parameterResult_'
savedParameterDir      = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_parameterResult_'+YYYY+'_'+mm+'_'+dd+'_'+HH+'_'+MM+'.json'
#savedParameterDir     = 'std_parameter.json'

#reading in images
    #img of sample
img = cv.imread(imgDir, cv.IMREAD_GRAYSCALE)
if img.any() == None: 
    print("Warning: No img could be loaded!\n")
    #img of background
imgBack = cv.imread(imgBackDir, cv.IMREAD_GRAYSCALE)
if imgBack.any() == None: 
    print("Warning: No imgBack could be loaded!\n")

#loading parameter for preprocessing and segmentation
fileLocalResponse = input("Do you want to load the parameters from selected file?\n(Otherwise the local parameters will be used) [y/n]: ")
if fileLocalResponse == 'y':
    readParameterFromFile(savedParameterDir) 
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
saveResponse = input("Do you want to store the result (+ parameters) [y/n]: ")
if saveResponse == 'y':
    now = datetime.now()
    dateTime = now.strftime("%Y_%m_%d_%H_%M")
    cv.imwrite(imgResultDir+dateTime+ext, img)
    writeParameterBackToFile(parameterResultDir+dateTime+'.json')
    print("Result and parameters are saved!\n")
elif saveResponse == 'n':
    print("Result and parameters are not saved!\n")
else:
    sys.exit("Wrong input!")
    