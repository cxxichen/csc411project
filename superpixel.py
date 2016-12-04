import numpy as np
import matplotlib.pyplot as plt
from utils import *

from skimage.segmentation import mark_boundaries
from skimage.feature import hog
# from sklearn import preprocessing
from skimage import exposure



def getSuperPixelLocations(superpixels):
    locations = []
    numSuperpixels = np.max(superpixels) + 1

    for i in xrange(0, numSuperpixels):
        indices = np.where(superpixels == i)
        x = np.mean(indices[0])
        y = np.mean(indices[1])
        locations.append([x, y])
    return np.array(locations)


def getSuperPixelMeanColor(image, superpixel):
    colors = []
    numSuperpixels = np.max(superpixel) + 1
    # newIm = np.ndarray(image.shape)

    for i in xrange(0, numSuperpixels):
        indices = np.where(superpixel == i)
        color = image[indices]
        r = np.mean(color[:, 0])
        g = np.mean(color[:, 1])
        b = np.mean(color[:, 2])
        colors.append([r, g, b])
        # newIm[indices] = [r,g,b]
    # plotImage(newIm)
    # plotSuperpixel(newIm, superpixel)
    return np.array(colors)


def getSuperPixelSize(superpixel):
    size = []
    numSuperpixels = np.max(superpixel) + 1
    print numSuperpixels
    for i in xrange(0, numSuperpixels):
        indices = np.where(superpixel == i)
        size.append(indices[0].shape)
    return np.array(size)


def getSuperPixelOrientedHistogram(superpixels, image):
    gradients = []
    numSuperpixels = np.max(superpixels) + 1
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02)) # 0 - 1

    for i in np.unique(superpixels):
    	indices = np.where(superpixels==i)
        gradient = [np.mean(hog_image_rescaled[indices])]
        gradients.append(gradient)
    return np.array(gradients)

def getSuperPixelShape(superpixels):
    shape = []
    numSuperpixels = np.max(superpixels) + 1
    for i in xrange(0,numSuperpixels):
        temp = np.zeros((1, 64),dtype = float)
        indices = np.where(superpixels == i)
        width = np.max(indices[0]) - np.min(indices[0])
        height = np.max(indices[1]) - np.min(indices[1])
        boxWidth = np.max((width, height))
        boundingBox = np.zeros((boxWidth + 1, boxWidth + 1))
        x = indices[0] - np.min(indices[0]) + boxWidth/2 - width/2
        y = indices[1] - np.min(indices[1]) + boxWidth/2 - height/2
        boundingBox[(x,y)] = 1
        boundingBox = boundingBox.ravel()
        binWidth = 1.0 * len(boundingBox) / 64
        for j in xrange(0,len(boundingBox)):
            if boundingBox[j] == 1:
                x = int(j/binWidth)
                temp[0][x] = temp[0][x] + 1
        shape.append(1.0 * temp[0] / np.sum(temp[0]))
    return np.array(shape)


def getSuperPixelLabel(image, superpixels, labelImage, thres):
    # newIm = image
    superpixel_labels = []
    numSuperpixels = np.max(superpixels) + 1
    labelValue = labelImage.max()
    label_pixels = (labelImage == labelValue)

    for i in xrange(0, numSuperpixels):
        indices = np.where(superpixels == i)
        cor_label = label_pixels[indices]
        portion_true = 1.0 * np.sum(cor_label) / len(cor_label)
        if portion_true > thres:
            superpixel_labels.append(1)
            # newIm[indices] = [1,1,1]
        else:
            superpixel_labels.append(0)
            # newIm[indices] = [0,0,0]
    # showPlots(newIm, numSuperpixels, superpixels)
    # show sample test mean image
    return np.array(superpixel_labels)

def getPairwiseMatrix(superpixels):
    numSuperpixels = np.max(superpixels) + 1
    row, col = superpixels.shape
    edges = np.zeros((numSuperpixels, numSuperpixels))
    for i in xrange(0, row - 1):
        for j in xrange(0, col - 1):
            if(superpixels[i][j] != superpixels[i][j+1]):
                edges[superpixels[i][j]][superpixels[i][j+1]] = 1
                edges[superpixels[i][j+1]][superpixels[i][j]] = 1
            if(superpixels[i][j] != superpixels[i+1][j]):
                edges[superpixels[i][j]][superpixels[i+1][j]] = 1
                edges[superpixels[i+1][j]][superpixels[i][j]] = 1
    return edges

def showPlots(image, numSuperpixels, superpixels):
    # show sample test mean image
    fig = plt.figure("Superpixels -- %d im_sp" % (numSuperpixels))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, superpixels))
    #bugged
    #ax.imsave("output/000%d.png" % (im_name),mark_boundaries(image, superpixels))
    plt.axis("off")
    plt.show()

def getPixelLabel(label_image):
    labelValue = label_image.max()
    label_pixels = (label_image == labelValue)
    return np.array(label_pixels)


def getSuperValidFiles(superpixels, count, valid_files):
    numSuperpixels = np.max(superpixels) + 1
    for i in xrange(0,numSuperpixels):
        valid_files.append(count)
    return valid_files
