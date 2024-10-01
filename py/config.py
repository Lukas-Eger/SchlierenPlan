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
        "y_start": 1265,
        "y_end": 1981,
        "x_start": 1427,
        "x_end": 2174,
       
        "setupContrast": True,
        "alpha": 3.2,  #lower contrast: alpha < 1, higher contrast alpha > 1 
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
        
        "deblureImage": True,
        "meanFilteredImage": False,
        "kernelSizeMeanFilter": 3,
        "gaussianFilteredImage": False,
        "kernelSizeGaussianFilter": 17,
        "standardDeviation": 0,         #calculated from kernelSize
        "medianFilteredImage": True,
        "kernelSizeMedianFilter": 9,
        "bilateralFilteredImage": False,
        "bilateralFilterDiameter": 31,  #diameter of each pixel neighborhood that is used during filtering
        "bilateralFilterSigma": 250,    #<10: no effect >150: huge effect on outcome
        
        "meanFilterFloodFill": False,
        "kernelMeanFloodFill": 21,
        "seedPointFloodFill": (0,0),
        "lowerTolFloodFill": 1,
        "upperTolFloodFill": 1,
        "kernelSizeOpening": 31,
        "kernelSizeClosing": 7,
        
        "doCannySegmentation": False,
        
        "doThresholdSegmentation": True,
        "adaptiveMethodThreshold": cv.ADAPTIVE_THRESH_GAUSSIAN_C,   #other mode: cv.ADAPTIVE_THRESH_MEAN_C
        "numberOfNeighborsThreshold": 21,
        "subtractedConstantThreshold": 2,
        "kernelSizeClosingThreshold": 5,
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
