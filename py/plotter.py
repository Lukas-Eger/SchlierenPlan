from matplotlib import pyplot as plt

def plotResults(img, imgCropped, imgContrast, imgEdgeFiltered, imgDeblured, maskFloodFill, maskOpening, maskNoForeground, maskClosing, imgSegmented):
    #intermediate results
    fig1, axs = plt.subplots(2,5)
    fig1.suptitle("intermediate results")
    
    #I. preprocessing
    axs[0, 0].set_ylabel("I. preprocessing")
    axs[0, 0].set_yticklabels([])
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])
    
    axs[0, 0].imshow(imgCropped, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].title.set_text("(3) shadow corrected\n(cropped) image")
    
    axs[0, 1].imshow(imgContrast, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].title.set_text("(4) contrasted\nimage")
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(imgEdgeFiltered, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].title.set_text("(5) edge filtered\nimage")
    axs[0, 2].axis('off')
    
    axs[0, 3].imshow(imgDeblured, cmap='gray', vmin=0, vmax=255)
    axs[0, 3].title.set_text("(6) deblured\nimage")
    axs[0, 3].axis('off')
    
    axs[0, 4].axis('off')
    
    #II. segmentation
    axs[1, 0].set_ylabel("II. segmentation")
    axs[1, 0].set_yticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xticks([])
    
    axs[1, 0].imshow(maskFloodFill, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].title.set_text("(0.1) mask after\nflood fill")
    
    axs[1, 1].imshow(maskOpening, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].title.set_text("(0.2) mask after\nopening")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(maskNoForeground, cmap='gray', vmin=0, vmax=255)
    axs[1, 2].title.set_text("(0.3) mask after\nremoving fg")
    axs[1, 2].axis('off')
    
    axs[1, 3].imshow(maskClosing, cmap='gray', vmin=0, vmax=255)
    axs[1, 3].title.set_text("(0.4) mask after\nclosing")
    axs[1, 3].axis('off')
    
    axs[1, 4].imshow(imgSegmented, cmap='gray', vmin=0, vmax=255)
    axs[1, 4].title.set_text("(1) segmented\nimage")
    axs[1, 4].axis('off')
    plt.show()
    
    #final result
    plt.figure()
    fig2 = plt.subplot()
    fig2.imshow(img, cmap = 'gray', vmin=0, vmax=255)
    plt.title("result")
    plt.axis('off')
    plt.show()
    