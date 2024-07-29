import cv2 as cv
import json

parameter = {
    "bgImgAvailable" : True,
    
    "doShadingCorrection": True,
    
    "cropImage": True,
    
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



with open('config.json', 'w') as json_file:
  json.dump(parameter, json_file)
  
  
with open('config.json', 'r') as f:
  parameter = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'French']}
print(data)
print(data==parameter)