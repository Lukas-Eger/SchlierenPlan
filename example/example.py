import sys
import os
import cv2 as cv
import json
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image, ImageEnhance

sys.path.append(os.path.abspath('../src'))
import SPImageProcessing as spip
from config import readParameterFromFile, writeParameterBackToFile, initLocalParameter


%matplotlib qt
#%matplotlib inline

plt.close("all")

#   img:                '../img/Tyy/xx_Tyy_zzzzzzz_img.ext'
#   imgBack:            '../img/Tyy/xx_Tyy_zzzzzzz_imgBack.ext'
#   imgResult:          '../img/Tyy/xx_Tyy_zzzzzzz_imgResult_YYYY_mm_dd_HH_MM.ext'
#   parameterResult:    '../img/Tyy/xx_Tyy_zzzzzzz_parameter_YYYY_mm_dd_HH_MM.json'
#--------- PLEASE CONFIGURE ------------
Tyy         = 'T03'         #number of test experiment yy
xx          = '01'          #number of image xx
zzzzzzz     = '82798'       #exposure time in us (unknown: 'zzzzzzz') or 'hdr' for High Dynamic Range Image
YYYY        = '2024'        #year YYYY
mm          = '10'          #month mm
dd          = '01'          #day dd
HH          = '23'          #hour HH
MM          = '02'          #minute MM
ext         = '.bmp'        #file extension: .bmp, .jpg, .png
#----------------------------------------
imgDir                 = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz+'_img'+ext
imgBackDir             = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz+'_imgBack'+ext
imgResultDir           = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz+'_imgResult_'
parameterResultDir     = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz+'_parameterResult_'
savedParameterDir      = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz+'_parameterResult_'+YYYY+'_'+mm+'_'+dd+'_'+HH+'_'+MM+'.json'
#savedParameterDir     = '../src/SPImageProcessing/std_parameter.json'

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
    
#1. PREPROCESSING
img, _, _, _, imgCropped, imgContrast, imgEdgeFiltered, imgDeblured = spip.preprocessing(img, imgBack)

#2. SEGMENTATION
img, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented = spip.segmentation(img)

#show (intermediate) results
spip.plotResults(img, imgCropped, imgContrast, imgEdgeFiltered, imgDeblured, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented)

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
    