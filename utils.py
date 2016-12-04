import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

def plotImage(image):
    plt.imshow(image)
    plt.show()

def plotGreyScale(image):
    plt.imshow(image * 256, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.show()

def plotSuperpixel(image, superpixel):
    plotImage(mark_boundaries(image, superpixel))
