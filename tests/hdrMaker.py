#see: https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

%matplotlib qt
#%matplotlib inline

plt.close("all")

#   img:                '../img/Tyy/xx_Tyy_zzzzzzz_iii.ext'
#--------- PLEASE CONFIGURE ------------
Tyy             = 'T04'         #number of test experiment yy
xx              = '01'          #number of image xx
zzzzzzz0        = '3000466'     #highest exposure time in us
zzzzzzz1        = '2000030'     #high exposure time in us
zzzzzzz2        = '800447'      #low exposure time in us
zzzzzzz3        = '309463'      #lowest exposure time in us
iii             = 'img'         #choose betweeng img or imgBack
ext             = '.bmp'        #file extension: .bmp, .jpg, .png

gammaDebevec    = 1.5
gammaRobertson  = 1.5
#----------------------------------------
img0Dir                 = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz0+'_'+iii+ext
img1Dir                 = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz1+'_'+iii+ext
img2Dir                 = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz2+'_'+iii+ext
img3Dir                 = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_'+zzzzzzz3+'_'+iii+ext
imgResultDir            = '../img/'+Tyy+'/'+xx+'_'+Tyy+'_hdr_'+iii+ext

#Loading exposure images + exposure times
imgLocations = [img0Dir, img1Dir, img2Dir, img3Dir]
imgList = [cv.cvtColor(cv.imread(loc), cv.COLOR_BGR2RGB) for loc in imgLocations]
exposureTime0 = np.float32(zzzzzzz0)/1000000.0
exposureTime1 = np.float32(zzzzzzz1)/1000000.0
exposureTime2 = np.float32(zzzzzzz2)/1000000.0
exposureTime3 = np.float32(zzzzzzz3)/1000000.0
exposureTimes = np.array([exposureTime0, exposureTime1, exposureTime2, exposureTime3], dtype=np.float32)

#Merge exposures to HDR image
#   1. Debavec
mergeDebevec = cv.createMergeDebevec()
hdrDebevec = mergeDebevec.process(imgList, times=exposureTimes.copy())
#tonemap1 = cv.createTonemap(gamma=gammaDebevec) #map data into the range [0..1]
#tonemap1 = cv.createTonemapDrago(gamma=gammaDebevec)
#tonemap1 = cv.createTonemapMantiuk(gamma=gammaDebevec)
tonemap1 = cv.createTonemapReinhard(gamma=gammaDebevec)
resDebevec = tonemap1.process(hdrDebevec.copy())
#   2. Robertson
mergeRobertson = cv.createMergeRobertson()
hdrRobertson = mergeRobertson.process(imgList, times=exposureTimes.copy())
#tonemap2 = cv.createTonemap(gamma=gammaRobertson)  #map data into the range [0..1]
#tonemap2 = cv.createTonemapDrago(gamma=gammaRobertson)
#tonemap2 = cv.createTonemapMantiuk(gamma=gammaRobertson)
tonemap2 = cv.createTonemapReinhard(gamma=gammaRobertson)
resRobertson = tonemap2.process(hdrRobertson.copy())
#   3. Mertens
mergeMertens = cv.createMergeMertens()
resMertens = mergeMertens.process(imgList)

#Convert datatype to 8-bit and save
imgDebevec8bit = np.clip(resDebevec * 255, 0, 255).astype('uint8')
imgRobertson8bit = np.clip(resRobertson * 255, 0, 255).astype('uint8')
imgMertens8bit = np.clip(resMertens * 255, 0, 255).astype('uint8')

#Plot images
fig1, axs = plt.subplots(2,4)
fig1.suptitle("Choose a hdr image!")

#input images
axs[0, 0].set_ylabel("input images")
axs[0, 0].set_yticklabels([])
axs[0, 0].set_xticklabels([])
axs[0, 0].set_yticks([])
axs[0, 0].set_xticks([])

axs[0, 0].imshow(imgList[0])
axs[0, 0].title.set_text('ET: '+zzzzzzz0+'us')

axs[0, 1].imshow(imgList[1])
axs[0, 1].title.set_text('ET: '+zzzzzzz1+'us')
axs[0, 1].axis('off')

axs[0, 2].imshow(imgList[2])
axs[0, 2].title.set_text('ET: '+zzzzzzz2+'us')
axs[0, 2].axis('off')

axs[0, 3].imshow(imgList[3])
axs[0, 3].title.set_text('ET: '+zzzzzzz3+'us')
axs[0, 3].axis('off')

#hdr images
axs[1, 0].set_ylabel("hdr images")
axs[1, 0].set_yticklabels([])
axs[1, 0].set_xticklabels([])
axs[1, 0].set_yticks([])
axs[1, 0].set_xticks([])

axs[1, 0].imshow(imgDebevec8bit)
axs[1, 0].title.set_text("1. Debevec")

axs[1, 1].imshow(imgRobertson8bit)
axs[1, 1].title.set_text("2. Robertson")
axs[1, 1].axis('off')

axs[1, 2].imshow(imgMertens8bit)
axs[1, 2].title.set_text("3. Mertens")
axs[1, 2].axis('off')

axs[1, 3].axis('off')
plt.show()

hdrResponse = input("Do you want to save a hdr image? [y/n]: ")
if hdrResponse == 'y':
    whichOneResponse = input("Which hdr image would you like to save? [1/2/3]: ")
    if whichOneResponse == '1':
        print("You saved a Debevec hdr image!\n")
        cv.imwrite(imgResultDir, cv.cvtColor(imgDebevec8bit, cv.COLOR_RGB2BGR))
    elif whichOneResponse == '2':
        print("You saved a Robertson hdr image!\n")
        cv.imwrite(imgResultDir, cv.cvtColor(imgRobertson8bit8bit, cv.COLOR_RGB2BGR))
    elif whichOneResponse == '3':
        print("You saved a Mertens hdr image!\n")
        cv.imwrite(imgResultDir, cv.cvtColor(imgMertens8bit, cv.COLOR_RGB2BGR))
    else:
        sys.exit("Wrong input!")
elif hdrResponse == 'n':
    print("You didn't save a hdr image!\n")
else:
    sys.exit("Wrong input!")