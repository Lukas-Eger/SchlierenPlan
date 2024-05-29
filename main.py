import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

mtx = np.array([[6578.6103146602, 0.0, 2005.963972956],
                [0.0, 6578.814113747, 1563.6107420985],
                [0.0, 0.0, 1.0]])

dist_k = [-0.2390637125, 0.2421873021, -0.2502353976, 0.0, 0.0, 0.0]
dist_p = [0.0011393958, 0.0003135942]
dist = np.array(dist_k[:2] + dist_p + dist_k[2:])

img = cv.imread('pic.jpg')

#destrot img
dst = cv.undistort(img, mtx, dist, None)

#rotate and cut
#rot=ndimage.rotate(dst, 2)
#rot=rot[100:900,50:170,:]
#cv.imshow(' Image', rot)
#cv.resizeWindow(' Image', 3000 , 3000)
#cv.waitKey(0)
#cv.destroyAllWindows()
gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(gray,-1,kernel)
#shading correction
shadcorimg=dst


#edge enhancement harris edge detection
edges = cv.Canny(shadcorimg,50,120)
plt.subplot(131),plt.imshow(rot,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.hist(rot.ravel(),256,[0,256])
# Save the undistorted image
cv.imwrite('calibresult.png', dst)