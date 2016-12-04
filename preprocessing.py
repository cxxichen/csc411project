import numpy as np
import glob
import os
import argparse
import scipy.io
import sys
import random
from ExtractFeature import *
from superpixel import *


fileNames = glob.glob("data_road/training/image_2/*.png")
labelFileNames = glob.glob("data_road/training/gt_image_2/*road*.png")
numImages = len(fileNames)

numTrain = int(numImages * 0.6)
numValid = int(numImages * 0.1)
numTest = numImages - (numTrain + numValid)


TRAINING = 0
VALIDATION = 1
TESTING = 2
fileLabels = np.zeros(numImages)
for i in xrange(0, numTest):
    fileLabels[i] = 2
for i in xrange(numTest, (numTest + numValid)):
    fileLabels[i] = 1

random.seed(100)
random.shuffle(fileLabels)

train_data = []
train_labels = []
valid_data = []
valid_labels = []
test_data = []
test_labels = []

# train_edges = []
# train_edgesFeatures1 = []
# train_edgesFeatures2 = []
# test_edges = []
# test_edgesFeatures1 = []
# test_edgesFeatures2 = []
# valid_edges = []
# valid_edgesFeatures1 = []
# valid_edgesFeatures2 = []

test_pixels_labels = []
valid_pixels_labels = []

valid_files = []
test_files = []

valid_files_count = 0
test_files_count = 0

train_superpixels = []
valid_superpixels = []
validation_original_image = []
test_superpixels = []
test_original_image = []

for i in xrange(0, numImages):
    feature = Feature()
    feature.loadImage(fileNames[i])

    feature.loadGreyScaleImage()

    feature.loadSuperPixel()

    feature.loadLabelImage(labelFileNames[i])

    labels = feature.getSuperPixelLabels()

    feature.extractFeature()
    featureVectors = feature.getFeaturesVectors()

    # edges, edgeFeatures1, edgeFeatures2 = feature.getEdges()

    if fileLabels[i] != TESTING:
		if fileLabels[i] == TRAINING:
			# train_edges.append(edges)
			# train_edgesFeatures1.append(edgeFeatures1)
			# train_edgesFeatures2.append(edgeFeatures2)
			train_superpixels.append(feature.getSuperpixelImage())
			train_labels = np.append(train_labels, labels, 0)
			if train_data == []:
				train_data = featureVectors
			else:
				train_data = np.vstack((train_data, featureVectors))
		else:
			# valid_edges.append(edges)
			# valid_edgesFeatures1.append(edgeFeatures1)
			# valid_edgesFeatures2.append(edgeFeatures2)
			valid_superpixels.append(feature.getSuperpixelImage())
			valid_labels = np.append(valid_labels, labels, 0)
			# print np.array(valid_labels).shape

			validation_original_image.append(fileNames[i])
			valid_files = getSuperValidFiles(feature.getSuperpixelImage(), valid_files_count, valid_files)
			valid_pixels_labels.append(getPixelLabel(feature.getLabelImage()))
			# print np.array(valid_pixels_labels).shape
			valid_files_count += 1
			if valid_data==[]:
				valid_data = featureVectors
			else:
				valid_data = np.vstack((valid_data, featureVectors))
    else:
		# test_files_count += 1
		# test_edges.append(edges)
		# test_edgesFeatures1.append(edgeFeatures1)
		# test_edgesFeatures2.append(edgeFeatures2)
		test_superpixels.append(feature.getSuperpixelImage())
		test_labels = np.append(test_labels, labels, 0)

		test_original_image.append(fileNames[i])
		test_files = getSuperValidFiles(feature.getSuperpixelImage(), test_files_count, test_files)
		test_pixels_labels.append(getPixelLabel(feature.getLabelImage()))
		test_files_count += 1

		if test_data==[]:
			test_data = featureVectors
		else:
			test_data = np.vstack((test_data, featureVectors))

    sys.stdout.write('\r')
    sys.stdout.write(fileNames[i] + " " + labelFileNames[i] + '\nprogress %2.2f%%' %(100.0*i/numImages))
    sys.stdout.flush()

scipy.io.savemat("train_matrices",
	{
		'train_data':train_data,
		'train_labels':train_labels,
		'valid_data':valid_data,
		'valid_labels':valid_labels,
		'file_labels':fileLabels,
		'im_file_names':fileNames,
		'label_file_names':labelFileNames,
		'valid_pixels_labels':valid_pixels_labels,
		'test_pixels_labels':test_pixels_labels
	},
	oned_as='column')

scipy.io.savemat("train_data",
	{
		'train_data':train_data,
		'train_labels':train_labels,
		'train_superpixels':train_superpixels
		# 'train_edges':train_edges,
		# 'train_edgesFeatures1':train_edgesFeatures1,
		# 'train_edgesFeatures2':train_edgesFeatures2
	},
	oned_as='column')

scipy.io.savemat("valid_data",
	{
		'valid_data':valid_data,
		'valid_labels':valid_labels,
		'valid_superpixels':valid_superpixels,
		# 'valid_edges':valid_edges,
		# 'valid_edgesFeatures1':valid_edgesFeatures1,
		# 'valid_edgesFeatures2':valid_edgesFeatures2,
		'valid_files':valid_files,
		'valid_files_count':valid_files_count,
		'validationOriginalImage':validation_original_image
	},
	oned_as='column')

scipy.io.savemat("test_data",
	{
		'test_data':test_data,
		'test_label':test_labels,
		'test_superpixels':test_superpixels,
        'test_files':test_files,
        'test_files_count':test_files_count,
        'testOriginalImage':test_original_image
		# 'test_edges':test_edges,
		# 'test_edgesFeatures1':test_edgesFeatures1,
		# 'test_edgesFeatures2':test_edgesFeatures2
	},
	oned_as='column')
