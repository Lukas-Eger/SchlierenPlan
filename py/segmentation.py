import cv2 as cv
from config import getParameter, setParameter

def segmentation(img):
    imgSegmented = None
    
    thresh2,_ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    thresh1= round(thresh2/2,0)
    img=cv.Canny(img, thresh1,thresh2)
    
    imgSegmented = img.copy()
    #print(thresh1,thresh2)
    
    return (img, imgSegmented)
    
