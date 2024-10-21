import sys
import cv2 as cv
import numpy as np
import json

parameter = None
imgDefault = np.full((50,50), 255)

def initLocalParameter():
    global parameter
#----------------------------------- PLEASE CONFIGURE -----------------------------------
    parameter = {
        "bgImgAvailable": True,
        
        "doImgCorrection": True,
        
        "doShadingCorrection": True,
        
        "cropImage": True,
        "y_start": 911,
        "y_end": 1578,
        "x_start": 1590,
        "x_end": 2407,
       
        "setupContrast": True,
        "alpha": 2.5,  #lower contrast: alpha < 1, higher contrast alpha > 1 
        "beta": -30,    #brightness -127 < beta < 127
        
        "edgeEnhancement": True,
        "laplacianFilteredImage": False,
        "ddepthLaplacian": cv.CV_8U,    #CV_8U, CV_16S, CV_32F, CV_64F
        "kernelSizeLaplacian": 7,
        "pillowSharpendImage": True,
        "factorSharpness": 7.0,
        "sobeFilteredImage": False,
        "kernelSizeSobelFilter": 7,
        "ddepthSobel": cv.CV_8U,        #CV_8U, CV_16S, CV_32F, CV_64F
        
        "deblureImage": False,
        "meanFilteredImage": False,
        "kernelSizeMeanFilter": 3,
        "gaussianFilteredImage": False,
        "kernelSizeGaussianFilter": 17,
        "standardDeviation": 0,         #calculated from kernelSize
        "medianFilteredImage": True,
        "kernelSizeMedianFilter": 3,
        "bilateralFilteredImage": False,
        "bilateralFilterDiameter": 31,  #diameter of each pixel neighborhood that is used during filtering
        "bilateralFilterSigma": 250,    #<10: no effect >150: huge effect on outcome
        
        "meanFilterFloodFill": False,
        "kernelMeanFloodFill": 21,
        "seedPointFloodFill": (0,0),
        "lowerTolFloodFill": 2,
        "upperTolFloodFill": 2,
        "kernelSizeOpening": 7,
        "kernelSizeClosing": 7,
        
        "doCannySegmentation": False,
        
        "doThresholdSegmentation": True,
        "adaptiveMethodThreshold": cv.ADAPTIVE_THRESH_GAUSSIAN_C,   #other mode: cv.ADAPTIVE_THRESH_MEAN_C
        "numberOfNeighborsThreshold": 125,
        "subtractedConstantThreshold": 2,
        "kernelSizeClosingThreshold": 2,
        "doClosingThreshold": True
    }
#----------------------------------------------------------------------------------------

def readParameterFromFile(filename):
    global parameter
    with open(filename, 'r') as std_p:
        parameter = json.load(std_p)
        if parameter == None:
            sys.exit("No parameters could be loaded!")
        
        
def writeParameterBackToFile(filename):
    global parameter
    with open(filename, 'w') as json_file:
        json.dump(parameter, json_file)
        
def getParameter():
    global parameter
    return parameter

def setParameter(p):
    global parameter
    parameter = p
