import cv2 as cv
import numpy as np
import json

parameter = None
imgDefault = np.full((50,50), 255)

#please edit 
def initLocalParameter():
    global parameter
    parameter = {
        "bgImgAvailable": True,
        
        "doImgCorrection": True,
        
        "doShadingCorrection": True,
        
        "cropImage": True,
        "x_start": 2042,
        "y_start": 821,
        "x_end": 2411,
        "y_end": 1212,
       
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
        "standardDeviation": 0,         #calculated from kernelSize
        "medianFilteredImage": True,
        "kernelSizeMedianFilter":9,
        "bilateralFilteredImage": False,
        "bilateralFilterDiameter": 31,  #diameter of each pixel neighborhood that is used during filtering
        "bilateralFilterSigma": 250,    #<10: no effect >150: huge effect on outcome
        
        "seedPointFloodFill": (0,0),
        "lowerTolFloodFill": 12,
        "upperTolFloodFill": 12,
        "kernelSizeOpening": 3,
        "kernelSizeClosing": 7,
        
        "doCannySegmentation": False,
        
        "doThresholdSegmentation": True,
        "adaptiveMethodThreshold": cv.ADAPTIVE_THRESH_GAUSSIAN_C,   #other mode: cv.ADAPTIVE_THRESH_MEAN_C
        "numberOfNeighborsThreshold": 15,
        "subtractedConstantThreshold": 2,
        "kernelSizeClosingThreshold": 3,
        "doClosingThreshold": True
    }

def readParameterFromFile(filename):
    global parameter
    with open(filename, 'r') as std_p:
        globals()['parameter'] = json.load(std_p)
        
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
