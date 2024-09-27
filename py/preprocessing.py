import numpy as np
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from matplotlib.widgets import RectangleSelector
from config import getParameter, setParameter

def preprocessing(img,imgBack):
    parameter = getParameter()
    
    #   1. image correction using camera matrix
    imgUndist = None
    imgBackUndist = None
    if parameter["doImgCorrection"]:
        mtx,dist,new_mtx = extractCameraParameter()
        img = cv.undistort(img,mtx,dist,None,new_mtx)
        imgUndist = img.copy()
        if parameter["bgImgAvailable"]:
            imgBack = cv.undistort(imgBack,mtx,dist,None,new_mtx)
            imgBackUndist = imgBack.copy()
    
    #   2. shading correction
    imgCorrectedShading = None
    if parameter["bgImgAvailable"] and parameter["doShadingCorrection"]:
    #   img = img.mean() * img./imgBack
        epsilon = 1e-10
        img = np.nan_to_num((np.mean(imgBack) * (img.astype(np.float64) / (imgBack.astype(np.float64) + epsilon))), nan=0.0, posinf=255, neginf=0).astype(np.uint8)
        imgCorrectedShading = img.copy()
    
    #   3. crop image
    imgCroppedImage = None
    if parameter["cropImage"]:
        cropStatus = input("Möchten Sie das Bild erneut ausschneiden? [y/n]")
        if(cropStatus=="y"):
            img=imageCropping(img)
        elif(cropStatus=="n"):
            #imgCroppedImage=img[]
            img=img[parameter["y_start"]:parameter["y_end"],parameter["x_start"]:parameter["x_end"]]        
        imgCroppedImage = img.copy()
        
    #   4. contrast image
    imgContrast = None
    if parameter["setupContrast"]:
        img = cv.convertScaleAbs(img, alpha=parameter["alpha"], beta=parameter["beta"])
        imgContrast = img.copy()
        
    #   5. edge enhancement with different approaches (perhaps after debluring image)
    imgEdgeFiltered = None
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

    #   6. deblure img with different approaches (perhaps before edge enhancement)
    imgDeblured= None
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
    return (imgUndist,imgBackUndist,imgCorrectedShading,imgCroppedImage,imgContrast,imgEdgeFiltered,imgEdgeFiltered,imgDeblured)

def imageCropping(image):
    parameter = getParameter()
    global x_start, y_start, x_end, y_end, cropping
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    
    oriImage = image.copy()

    def mouse_crop(event, x, y, flags, param):
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
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv.imshow("Cropped", roi)

    cv.destroyAllWindows()
    cv.namedWindow("image")
    cv.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        if cropping:
            cv.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv.imshow("image", i)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to exit the loop
            break

    cv.destroyAllWindows()
    
    parameter["x_start"] = x_start
    parameter["x_end"] = x_end
    parameter["y_start"] = y_start
    parameter["y_end"] = y_end
    
    setParameter(parameter)
    
    return oriImage[y_start:y_end, x_start:x_end]

def extractCameraParameter():
    calib_df = pd.read_json('Alvium1800U-1240c_calib_df.json')
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
