import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matplotlib.widgets import RectangleSelector
import cv2
import numpy as np
def extractCameraParameter(calib_df):
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




def preprocessing(img,imgBack ,parameter):
        
    #pre-processing pipline
    #   1. shading correction
    imgCorrectedShading = None
    if parameter["bgImgAvailable"] and parameter["doShadingCorrection"]:
    #   img = img.mean() * img./imgBack
        epsilon = 1e-10
        img = np.nan_to_num((np.mean(imgBack) * (img.astype(np.float64) / (imgBack.astype(np.float64) + epsilon))), nan=0.0, posinf=255, neginf=0).astype(np.uint8)
        imgCorrectedShading = img.copy()
    
    #   2. crop image
    imgCroppedImage = None
    if parameter["cropImage"]:
        #implementation with help of PIL module
        img=imageCropping(img)
        imgCroppedImage = img.copy()
        
    #   3. contrast image
    imgContrast = None
    if parameter["setupContrast"]:
        img = cv.convertScaleAbs(img, alpha=parameter["alpha"], beta=parameter["beta"])
        imgContrast = img.copy()
        
    #   x. edge enhancement with different approaches (perhaps after debluring image)
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

    #   x. deblure img with different approaches (perhaps before edge enhancement)
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
    return (imgCorrectedShading,imgCroppedImage,imgContrast,imgEdgeFiltered,imgEdgeFiltered,imgDeblured)


def imageCropping(image):
    
    global x_start, y_start, x_end, y_end, cropping
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    oriImage = image.copy()

    def mouse_crop(event, x, y, flags, param):
        global x_start, y_start, x_end, y_end, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                x_end, y_end = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False
            refPoint = [(x_start, y_start), (x_end, y_end)]
            if len(refPoint) == 2:
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)

    cv2.destroyAllWindows()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        if cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to exit the loop
            break

    cv2.destroyAllWindows()
    return oriImage[y_start:y_end, x_start:x_end]
def imageCropping(img):
    bild=img.copy() 
    return bild[700:1361,2000:2500]
"""
done = False
def CropImage(img):

    
    # Callback for rectangle selection
    def select_callback(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        print(f"({x1}, {y1}) --> ({x2}, {y2})")
        print(f"The buttons you used were: {eclick.button} {erelease.button}")
        
        # Crop the image based on the selected coordinates
        cropped_img = img[y1:y2, x1:x2]
        plt.figure()
        plt.imshow(cropped_img, cmap='gray')
        plt.title('Cropped Image')
        plt.show()
    
    def toggle_selector(event):
        print('Key pressed.')
        if event.key == 't':
            if selector.active:
                print('RectangleSelector deactivated.')
                selector.set_active(False)
            else:
                print('RectangleSelector activated.')
                selector.set_active(True)
    
    def onselect(eclick, erelease):
        done = True
    # Load the image
    
    
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Click and drag to draw a rectangle.\nPress 't' to toggle the selector on and off.")
    
    # Create a RectangleSelector
    selector = RectangleSelector(
        ax, select_callback,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)
    
    # Connect the toggle function to key press event
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    
    plt.show()
    
    while (True):
        if done:
            break
    return cropped_img

"""
