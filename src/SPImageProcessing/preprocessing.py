import sys
import os
import numpy as np
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from matplotlib.widgets import RectangleSelector

current_dir = os.path.dirname(__file__)
example_path = os.path.abspath(os.path.join(current_dir, '../../example'))
sys.path.append(example_path)
from config import getParameter, setParameter, imgDefault

def preprocessing(img, imgBack):
    global imgDefault
    parameter = getParameter()
    
    #   1.1 Undistort image
    imgUndist = None
    imgBackUndist = None
    if parameter["doImgCorrection"]:
        mtx,dist, new_mtx = extractCameraParameter()
        img = cv.undistort(img, mtx, dist, None, new_mtx)
        imgUndist = img.copy()
        if parameter["bgImgAvailable"]:
            imgBack = cv.undistort(imgBack, mtx, dist, None, new_mtx)
            imgBackUndist = imgBack.copy()
    
    #   1.2 Shading correction
    imgCorrectedShading = imgDefault
    if parameter["bgImgAvailable"] and parameter["doShadingCorrection"]:
    #   img = imgBack.mean()*img/imgBack
        imgBackMean = np.mean(imgBack)
        imgBack_ = np.where(imgBack > 0, imgBack, 1)
        imgCorrectedShading = ((imgBackMean.astype(np.float64) * img.astype(np.float64)) / imgBack_.astype(np.float64)).astype(np.uint8)
        cv.normalize(imgCorrectedShading, imgCorrectedShading, 0, 255, cv.NORM_MINMAX)
        img = imgCorrectedShading.copy()
    
    #   1.3 Cropping image
    imgCroppedImage = imgDefault
    if parameter["cropImage"]:
        croppedResponse = input("Would you like to crop the image (again)?\n(Otherwise, the coordinates from the parameters are used.) [y/n]: ")
        if(croppedResponse == "y"):
            img = imageCropping(img)
        elif(croppedResponse == "n"):
            img = img[parameter["y_start"]:parameter["y_end"], parameter["x_start"]:parameter["x_end"]]     
        else:
            sys.exit("Wrong input!")
        print("cutout coordinates: y_start =", parameter["y_start"], "; y_end =", parameter["y_end"], "; x_start =", parameter["x_start"], "; x_end =", parameter["x_end"], "\n")
        imgCroppedImage = img.copy()
        
    #   1.4 Setup contrast
    imgContrast = imgDefault
    if parameter["setupContrast"]:
        img = cv.convertScaleAbs(img, alpha=parameter["alpha"], beta=parameter["beta"])
        imgContrast = img.copy()
        
    #   1.5 Edge enhancement
    imgEdgeFiltered = imgDefault
    if parameter["edgeEnhancement"]:
    #       1 laplacian filter
            imgLaplacianFiltered = cv.Laplacian(img, parameter["ddepthLaplacian"], ksize=parameter["kernelSizeLaplacian"])
            cv.normalize(imgLaplacianFiltered, imgLaplacianFiltered, 0, 255, cv.NORM_MINMAX)
    #       2 from pillow: image enhancer
            img_pil = Image.fromarray(img)
            enhancer = ImageEnhance.Sharpness(img_pil)
            imgPillowSharpend = enhancer.enhance(parameter["factorSharpness"])
            imgPillowSharpend = np.array(imgPillowSharpend)
    #       3 sobel filter
            imgSobelFiltered = cv.Sobel(img, ddepth=parameter["ddepthSobel"], dx=1, dy=1, ksize=parameter["kernelSizeSobelFilter"])
            cv.normalize(imgSobelFiltered, imgSobelFiltered, 0, 255, cv.NORM_MINMAX)
            if parameter["laplacianFilteredImage"]:
                img = imgLaplacianFiltered.copy()
            elif parameter["pillowSharpendImage"]:
                img = imgPillowSharpend.copy()
            elif parameter["sobeFilteredImage"]:
                img = imgSobelFiltered.copy()
            else:
                print("Warning: No edge enhancement was chosen although active\n")
            imgEdgeFiltered= img.copy()

    #   1.6 Denoise image
    imgDeblured= imgDefault
    if parameter["deblureImage"]:
    #       1 mean filter:
            imgMeanFiltered = cv.blur(img,(parameter["kernelSizeMeanFilter"],parameter["kernelSizeMeanFilter"]))
    #       2 gaussian filter
            imgGaussianFiltered = cv.GaussianBlur(img, (parameter["kernelSizeGaussianFilter"],parameter["kernelSizeGaussianFilter"]), parameter["standardDeviation"])
    #       3 median filter
            imgMedianFiltered = cv.medianBlur(img, ksize=parameter["kernelSizeMedianFilter"])
    #       4 Bilateral Filtering (keeping edges sharp)
            imgBilateralFiltered = cv.bilateralFilter(img, parameter["bilateralFilterDiameter"], parameter["bilateralFilterSigma"], parameter["bilateralFilterSigma"])
            if parameter["meanFilteredImage"]:
                img = imgMeanFiltered.copy()
            elif parameter["gaussianFilteredImage"]:
                img = imgGaussianFiltered.copy()
            elif parameter["medianFilteredImage"]:
                img = imgMedianFiltered.copy()
            elif parameter["bilateralFilteredImage"]:
                img = imgBilateralFiltered.copy()
            else:
                print("Warning: No debluring Filter was chosen although active\n")
            imgDeblured = img.copy()
    return (img, imgUndist, imgBackUndist, imgCorrectedShading, imgCroppedImage, imgContrast, imgEdgeFiltered, imgDeblured)


def imageCropping(img):
    parameter = getParameter()
    
    global x_start, y_start, x_end, y_end, cropping
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    
    imgOriginal = img.copy()

    def mouseCropping(event, x, y, flags, param):
        global x_start, y_start, x_end, y_end, cropping
        if event == cv.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
        elif event == cv.EVENT_MOUSEMOVE:
            if cropping:
                x_end, y_end = x, y
        elif event == cv.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False
            refPoint = [(x_start, y_start), (x_end, y_end)]
            if len(refPoint) == 2:
                imgCropped = imgOriginal[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv.imshow("cropped image (press 'q' to continue)", imgCropped)

    cv.destroyAllWindows()
    cv.namedWindow("uncropped image (draw a rectangle from the top left to the bottom right corner)")
    cv.setMouseCallback("uncropped image (draw a rectangle from the top left to the bottom right corner)", mouseCropping)

    while True:
        i = img.copy()
        if cropping:
            cv.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv.imshow("uncropped image (draw a rectangle from the top left to the bottom right corner)", i)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    
    #save coordinates for cutout
    parameter["x_start"] = x_start
    parameter["x_end"] = x_end
    parameter["y_start"] = y_start
    parameter["y_end"] = y_end
    setParameter(parameter)
    
    return imgOriginal[y_start:y_end, x_start:x_end]


def extractCameraParameter():
    calib_df = pd.read_json('../src/SPImageProcessing/Alvium1800U-1240c_calib_df.json')
    mtx = np.array(calib_df.at['camera_matrix', 'values'])
    new_mtx = np.array(calib_df.at['new_camera_matrix', 'values'])
    roi = np.array(calib_df.at['roi', 'values'])
    dist_k = calib_df.at['dist_k', 'values']
    dist_p = calib_df.at['dist_p', 'values']
    dist_s = calib_df.at['dist_s', 'values']
    dist_tau = calib_df.at['dist_tau', 'values']
    dist = np.array([dist_k[0], dist_k[1], dist_p[0], dist_p[1], dist_k[2], dist_k[3], dist_k[4], dist_k[5],
                    dist_s[0], dist_s[1], dist_s[2], dist_s[3], dist_tau[0], dist_tau[1]])
    return (mtx,dist,new_mtx)
