import numpy as np
import matplotlib.pyplot as plt
from utils import *
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io, color

import superpixel as sp

N_SEGMENTS = 500
print "N_SEGMENTS: %s" % N_SEGMENTS
COMPACTNESS = 30
print "COMPACTNESS: %s" % COMPACTNESS
SIGMA = 1

class Feature:

    def __inin__(self):
        self.image = []
        self.imageGreyScale = []
        self.superpixel = []
        self.labelImage = []
        self.superpixelLocation = []
        self.superpixelColor = []
        self.superpixelSize = []
        self.superpixelHog = []
        self.featureVectors = []
        self.superpixelLabels = []
        self.edges = []
        self.edge_featureVectors = []

    def loadImage(self, imgUrl):
        self.image = img_as_float(io.imread(imgUrl))
        # plotImage(self.image)

    def loadGreyScaleImage(self):
        self.imageGreyScale = color.rgb2gray(self.image)
        # plotGreyScale(self.imageGreyScale)

    def loadSuperPixel(self):
        if self.image == []:
            raise Exception("Load Image before getting superpixel")
        else:
            self.superpixel = slic(
                self.image,
                n_segments=N_SEGMENTS,
                compactness=COMPACTNESS,
                sigma=SIGMA)
        # plotSuperpixel(self.image, self.superpixel)

    def loadLabelImage(self, labelUrl):
        self.labelImage = color.rgb2gray(io.imread(labelUrl))
        # plotImage(self.labelImage)
        # self.superpixelLabels = sp.getSuperPixelLabel(self.superpixel, self.label_image, 0.5)

    def extractFeature(self):
        self.superpixelLocation = sp.getSuperPixelLocations(self.superpixel)
        # print self.superpixelLocation.shape

        self.superpixelColor = sp.getSuperPixelMeanColor(self.image, self.superpixel)
        # print self.superpixelColor.shape

        self.superpixelSize = sp.getSuperPixelSize(self.superpixel)
        # print self.superpixelSize.shape

        self.superpixelHog = sp.getSuperPixelOrientedHistogram(self.superpixel, self.imageGreyScale)
        # print self.superpixelHog.shape

        # self.superpixelShape = sp.getSuperPixelShape(self.superpixel)
        # print self.superpixelShape.shape

        self.featureVectors = np.vstack((self.superpixelLocation.T,
            self.superpixelColor.T,
            self.superpixelHog.T,
            self.superpixelSize.T)).T

        # self.featureVectors = np.vstack((self.superpixelLocation.T,
        #     self.superpixelColor.T,
        #     self.superpixelSize.T)).T

        # self.featureVectors = np.vstack((self.superpixelLocation.T,
        #     self.superpixelColor.T,
        #     self.superpixelHog.T,
        #     self.superpixelSize.T,
        #     self.superpixelShape.T)).T


    def getEdges(self):
        #
        self.edges = sp.getPairwiseMatrix(self.superpixel)
        row = col = np.max(self.superpixel) + 1
        sumDiff = 0
        count = 0
        self.edgesGrad = np.zeros((row, col))
        self.edgesDist = np.zeros((row, col))

        for i in xrange(0, row):
            for j in xrange(i, col):
                if self.edges[i][j] != 0:
                    self.edgesGrad[i][j] = np.linalg.norm(self.superpixelColor[i] - self.superpixelColor[j])
                    self.edgesGrad[j][i] = self.edgesGrad[i][j]

                    self.edgesDist[i][j] = np.linalg.norm(self.superpixelLocation[i] - self.superpixelLocation[j])
                    self.edgesDist[j][i] = self.edgesDist[i][j]
                    #self.edge_featureVectors.append([self.edgesGrad[i][j], self.edgesDist[i][j]])
                    sumDiff += self.edgesGrad[i][j]
                    count += 1
        expectColorGrad = sumDiff/count
        #print self.edge_featureVectors[:][0]
        #print expectColorGrad
        return self.edges, self.edgesGrad, self.edgesDist


    def getSuperPixelLabels(self):
        self.superpixelLabels = sp.getSuperPixelLabel(self.image, self.superpixel, self.labelImage, 0.5)
        return self.superpixelLabels

    def getSuperpixelsLocation(self):
        return self.superpixelLocation

    def getSuperpixelsColor(self):
        return self.superpixelColor

    def getSuperpixelsHog(self):
        return self.superpixelHog

    def getSuperpixelsSize(self):
        return self.superpixelSize

    def getFeaturesVectors(self):
        return self.featureVectors

    def getImage(self):
        return self.image

    def getSuperpixelImage(self):
        return self.superpixel

    def getLabelImage(self):
        return self.labelImage
