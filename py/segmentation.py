import cv2 as cv
import numpy as np
from config import getParameter, setParameter, imgDefault

def segmentation(img):
    global imgDefault
    parameter = getParameter()
    
    #   0. create mask for background
    height, width = img.shape[:2]
    #       .1 flood fill
    maskFloodFill = img.copy()
    if parameter["meanFilterFloodFill"]:
        maskFloodFill = cv.blur(maskFloodFill,(parameter["kernelMeanFloodFill"],parameter["kernelMeanFloodFill"]))
    m = np.zeros((height + 2, width + 2), np.uint8)
    cv.floodFill(maskFloodFill, m, parameter["seedPointFloodFill"], 0, parameter["lowerTolFloodFill"], parameter["upperTolFloodFill"])
    #       .2 opening (erosion) "to remove small white regions"
    kernelOpening = np.ones((parameter["kernelSizeOpening"], parameter["kernelSizeOpening"]), np.uint8)
    maskOpening = cv.morphologyEx(maskFloodFill, cv.MORPH_OPEN, kernelOpening)
    #       .3 remove foreground "every pixel that does not have the value 0 becomes white"
    maskNoForeground = np.where(maskOpening > 0, 255, maskOpening)
    #       .4 closing (dilation) "to remove small black regions"
    kernelClosing = np.ones((parameter["kernelSizeClosing"], parameter["kernelSizeClosing"]), np.uint8)
    maskClosing = cv.morphologyEx(maskNoForeground, cv.MORPH_CLOSE, kernelClosing)
    
    #   1. segmentation
    imgSegmented = imgDefault
    if parameter["doCannySegmentation"]:
        thresh2,_ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
        thresh1 = round(thresh2/2, 0)
        img = cv.Canny(img, thresh1, thresh2)
    elif parameter["doThresholdSegmentation"]:
        img = cv.adaptiveThreshold(img, 255, parameter["adaptiveMethodThreshold"], cv.THRESH_BINARY, parameter["numberOfNeighborsThreshold"], parameter["subtractedConstantThreshold"])
        if parameter["doClosingThreshold"]:
            kernelEllipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (parameter["kernelSizeClosingThreshold"], parameter["kernelSizeClosingThreshold"]))
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernelEllipse)
    else:
        print("No segmentation algorithm is selected!\n")
     
    #apply background mask
    img = cv.bitwise_and(img, img, mask=maskClosing)
    imgSegmented = img.copy()
    
    return (img, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented)
    