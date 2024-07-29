import cv2 as cv
def segmentation(img,parameter):
    thresh2,_ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    thresh1= round(thresh2/2,0)
    re=cv.Canny(img, thresh1,thresh2)
    print(thresh1,thresh2)
    return  re
    
