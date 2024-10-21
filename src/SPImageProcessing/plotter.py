import sys
import os
import numpy as np
from matplotlib import pyplot as plt

current_dir = os.path.dirname(__file__)
example_path = os.path.abspath(os.path.join(current_dir, '../../example'))
sys.path.append(example_path)

from config import getParameter, setParameter, imgDefault


def plotResults(img, imgColored, imgCropped, imgContrast, imgEdgeFiltered, imgDeblured, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented):
    parameter = getParameter()
    
    #intermediate results
    fig1, axs = plt.subplots(2,5)
    fig1.suptitle("intermediate results")
    
    #I. preprocessing
    axs[0, 0].set_ylabel("1. PREPROCESSING")
    axs[0, 0].set_yticklabels([])
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])
    
    axs[0, 0].imshow(imgCropped, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].title.set_text("1.3 Shadow corrected\n(cropped) image")
    
    axs[0, 1].imshow(imgContrast, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].title.set_text("1.4 Contrasted\nimage")
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(imgEdgeFiltered, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].title.set_text("1.5 Edge enhanced\nimage")
    axs[0, 2].axis('off')
    
    axs[0, 3].imshow(imgDeblured, cmap='gray', vmin=0, vmax=255)
    axs[0, 3].title.set_text("1.6 Denoised\nimage")
    axs[0, 3].axis('off')
    
    axs[0, 4].axis('off')
    
    #II. segmentation
    axs[1, 0].set_ylabel("2. SEGMENTATION")
    axs[1, 0].set_yticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xticks([])
    
    axs[1, 0].imshow(maskFloodFill, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].title.set_text("2.1.1 Mask after\nflood fill")
    
    axs[1, 1].imshow(maskOpening, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].title.set_text("2.1.2 Mask after\nopening")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(maskNoForeground, cmap='gray', vmin=0, vmax=255)
    axs[1, 2].title.set_text("2.1.3 Mask after\nremoving fg")
    axs[1, 2].axis('off')
    
    axs[1, 3].imshow(maskClosing, cmap='gray', vmin=0, vmax=255)
    axs[1, 3].title.set_text("2.1.4 Mask after\nclosing")
    axs[1, 3].axis('off')
    
    axs[1, 4].imshow(imgSegmented, cmap='gray', vmin=0, vmax=255)
    axs[1, 4].title.set_text("2.3 Segmented\nimage")
    axs[1, 4].axis('off')
    plt.show()
    
    #final result
    fig2, axs2 = plt.subplots(1,2)
    fig2.suptitle("result")
    axs2[0].imshow(imgColored[parameter["y_start"]:parameter["y_end"],parameter["x_start"]:parameter["x_end"]])
    axs2[0].title.set_text("captured\nshadow image")
    axs2[0].axis('off')
    axs2[1].imshow(imgSegmented, cmap='gray', vmin=0, vmax=255)
    axs2[1].title.set_text("image after\nsegmentation")
    axs2[1].axis('off')
    plt.show()
    