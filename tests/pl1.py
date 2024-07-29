"""
Pipline for Digital Image Processing

pl1: Pipline 1

"""

import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.image as matimg
from datetime import datetime
filename='../img/V1/h_std_05_raw.jpg'
img = cv.imread(filename)
#SW Color
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gray = cv.medianBlur(gray,9)
#gray=cv.fastNlMeansDenoising(gray,None,10,10,30)
#Average Filter



cutout=gray.copy()
ret,thresh1 = cv.threshold(cutout,160,255,cv.THRESH_BINARY)
cv.floodFill(thresh1, None, (0,0), 0)
cutout[np.where(thresh1==0)]=0

n=7
kernel = np.ones((n,n),np.float32)/(n*n)
average = cv.filter2D(cutout,-1,kernel)


#edges = cv.Canny(gray,50,120)
#laplace kantenfilter
dst = cv.Laplacian(average, cv.CV_16S, ksize=11)
#normailisieren
cv.normalize(dst,  dst, 0, 255, cv.NORM_MINMAX)

#edgecut=edges.copy()
#edgecut[np.where(thresh1==0)]=255
n=30
kernel = np.ones((n,n),np.float32)/(n*n)
dst_blurr = cv.filter2D(dst,-1,kernel)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(241),plt.hist(gray.ravel(),256,[0,256])
plt.subplot(242),plt.imshow(gray,cmap = 'gray')
plt.subplot(243),plt.imshow(average,cmap = 'gray')
plt.subplot(244),plt.imshow(cutout,cmap = 'gray')
#plt.subplot(243), plt.imshow(thresh1, cmap="gray")
plt.subplot(245),plt.imshow(dst,cmap = 'gray')
#plt.subplot(141),plt.hist(cutout.ravel(),256,[0,256])
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
cv.imwrite((filename+date_time+'_edges'+'.png'), dst)
#cv.imwrite((filename+date_time+'_edges_blurr'+'.png'), dst_blurr)
#plt.figure()
#plt.imshow(dst,cmap = 'gray')
#plt.savefig(('edges'+date_time+'.png'))

