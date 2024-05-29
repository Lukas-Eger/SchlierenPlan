import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

img = cv.imread('pic.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gray = cv.medianBlur(gray,9)
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(gray,-1,kernel)



cutout=gray.copy()
ret,thresh1 = cv.threshold(cutout,50,255,cv.THRESH_BINARY)
cv.floodFill(thresh1, None, (0,0), 0)
cutout[np.where(thresh1==0)]=180


#edges = cv.Canny(gray,50,120)
dst = cv.Laplacian(cutout, cv.CV_16S, ksize=11)
#edgecut=edges.copy()
#edgecut[np.where(thresh1==0)]=255

plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(gray,cmap = 'gray')
plt.subplot(144),plt.imshow(dst,cmap = 'gray')
plt.subplot(141),plt.hist(gray.ravel(),256,[0,256])
plt.subplot(143),plt.imshow(cutout,cmap = 'gray')
#plt.subplot(143), plt.imshow(thresh1, cmap="gray")
#plt.subplot(141),plt.hist(cutout.ravel(),256,[0,256])



