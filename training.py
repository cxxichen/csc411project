import matplotlib.pyplot as plt
import numpy as np
from utils import *
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy import misc
from ExtractFeature import *
from superpixel import *
from data_read import *
from skimage.segmentation import mark_boundaries

from sklearn import svm, metrics, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import time
from sklearn.preprocessing import StandardScaler

import benchmark as bm
from skimage import io, color

class road_estimation:
	def __init__(self, model_selection):
		self._train_data, self._train_targets, self._valid_data, \
		self._valid_targets, self._test_data, self._test_targets = data_load()
		self.valid_files_count, self.valid_files, self.valid_superpixels, self.valid_pixels_labels = load_valid_file()
		self.test_files_count, self.test_files, self.test_superpixels, self.test_pixels_labels = load_test_file()

		print self._train_data.shape

		self._model_selection = model_selection
		self._classifier = []

	def train(self):
		if self._model_selection == "svm":
			start = time.clock()
			self._classifier = svm.SVC(kernel='rbf', probability=False)
		elif self._model_selection == "nb":
			start = time.clock()
			self._classifier = GaussianNB()
		elif self._model_selection == "knn":
			k = 21
			print "k = %s" % k
			start = time.clock()
			self._classifier = KNeighborsClassifier(n_neighbors=k)
		elif self._model_selection == "ada":
			ne = 110
			print "n_estimators: %s" % ne
			start = time.clock()
			self._classifier = AdaBoostClassifier(n_estimators=ne)
		elif self._model_selection == "rf":
			ne = 50
			print "n_estimators: %s" % ne
			start = time.clock()
			self._classifier = RandomForestClassifier(n_estimators=ne)
		elif self._model_selection == "qda":
			start = time.clock()
			self._classifier = QuadraticDiscriminantAnalysis()
		elif self._model_selection == "nn":
			num_hidden_1 = 80
			num_hidden_2 = 150
			print "num_hidden_1:" + repr(num_hidden_1) + '\n' + "num_hidden_2:" + repr(num_hidden_2)
			start = time.clock()
			self._classifier = MLPClassifier(
				hidden_layer_sizes=(num_hidden_1, num_hidden_2),
				# hidden_layer_sizes=num_hidden_1,
				activation='relu',
				solver='sgd',
				alpha=0.01,
				batch_size=100,
				learning_rate='constant',
				learning_rate_init=0.01,
				max_iter=500, early_stopping=False, random_state=10)
		elif self._model_selection == "dt":
			md = 8
			print "Max Depth: %s" % md
			start = time.clock()
			self._classifier = tree.DecisionTreeClassifier(max_depth=md)
		elif self._model_selection == "bag":
			ne = 50
			print "n_estimators: %s" % ne
			start = time.clock()
			self._classifier = BaggingClassifier(n_estimators=ne)
		else:
			print "Please refer to one classifier"

		self.scaler = StandardScaler()
		self.scaler.fit(self._train_data)
		self._train_data = self.scaler.transform(self._train_data)
		self._classifier.fit(self._train_data, self._train_targets)

		self._valid_data = self.scaler.transform(self._valid_data)

		end = time.clock()
		print "Training time : %.4f second" % (end - start)

		prediction_valid = self._classifier.predict(self._valid_data)

		# print validation result for selected model.
		print("Classification report for classifier %s on valid_data:\n%s\n"
	      % (self._model_selection, metrics.classification_report(self._valid_targets, prediction_valid, digits=4)))

		print("Log_loss: %s"
	      % (metrics.log_loss(self._valid_targets, prediction_valid)))


		# superpixelTotal = pixelTotal = superpixelCorrect = pixelCorrect = 0
		#
		# for file_num in range(0, self.valid_files_count):
		# 	temp1, temp2 = bm.accuracyOfSuperpixels(file_num, self.valid_files, self._valid_data, self._classifier, self._valid_targets)
		# 	temp3, temp4 = bm.accuracyOfPixels(file_num,self.valid_files, self.valid_superpixels, self._valid_data, self._classifier, self.valid_pixels_labels)
		# 	superpixelCorrect = superpixelCorrect + temp1
		# 	superpixelTotal = superpixelTotal + temp2
		# 	pixelCorrect = pixelCorrect + temp3
		# 	pixelTotal = pixelTotal +temp4
		# bm.overrallAverageResult(superpixelCorrect, superpixelTotal, pixelCorrect, pixelTotal)


	def test(self):
		self._test_data = self.scaler.transform(self._test_data)
		prediction_test = self._classifier.predict(self._test_data)
		# print test result for selected model.
		print("Classification report for classifier %s on test_data:\n%s\n"
			% (self._model_selection, metrics.classification_report(self._test_targets, prediction_test, digits=4)))

		superpixelTotal = pixelTotal = superpixelCorrect = pixelCorrect = 0

		for file_num in range(0, self.test_files_count):
			temp1, temp2 = bm.accuracyOfSuperpixels(file_num, self.test_files, self._test_data, self._classifier, self._test_targets)
			temp3, temp4 = bm.accuracyOfPixels(file_num,self.test_files, self.test_superpixels, self._test_data, self._classifier, self.test_pixels_labels)
			superpixelCorrect = superpixelCorrect + temp1
			superpixelTotal = superpixelTotal + temp2
			pixelCorrect = pixelCorrect + temp3
			pixelTotal = pixelTotal +temp4
		bm.overrallAverageResult(superpixelCorrect, superpixelTotal, pixelCorrect, pixelTotal)

	def showPredictionImage(self):
		# show prediction image
		feature = Feature()
		feature.loadImage("um_000093.png")
		feature.loadLabelImage("um_road_000093.png")
		feature.loadSuperPixel()
		feature.loadGreyScaleImage()
		feature.extractFeature()
		fea_matrix = feature.getFeaturesVectors()
		fea_matrix = self.scaler.transform(fea_matrix)
		predict = self._classifier.predict(fea_matrix)

		image = np.copy(feature.image)
		num_superpixels = np.max(feature.superpixel) + 1
		for i in xrange(0, num_superpixels):
			indices = np.where(feature.superpixel == i)
			if predict[i] == 1:
				image[indices[0],indices[1],0] = 0
				image[indices[0],indices[1],1] = 0
				image[indices[0],indices[1],2] = 1
			if predict[i] == 0:
				image[indices[0],indices[1],0] = 1
				image[indices[0],indices[1],1] = 0
				image[indices[0],indices[1],2] = 0

		labelImage = feature.getLabelImage()
		labelImage_rgb = color.gray2rgb(labelImage)
		v = np.max(labelImage)
		for i in xrange(0, labelImage.shape[0]):
			for j in xrange(0, labelImage.shape[1]):
				if labelImage[i, j] == v:
					labelImage_rgb[i][j][0] = 0
					labelImage_rgb[i][j][1] = 0
					labelImage_rgb[i][j][2] = 1
				else:
					labelImage_rgb[i][j][0] = 1
					labelImage_rgb[i][j][1] = 0
					labelImage_rgb[i][j][2] = 0

		diff = image - labelImage_rgb
		for i in xrange(0, labelImage.shape[0]):
			for j in xrange(0, labelImage.shape[1]):
				if diff[i][j][0] == 0 and diff[i][j][1] == 0 and diff[i][j][2] == 0:
					diff[i][j][0] = 1
					diff[i][j][1] = 1
					diff[i][j][2] = 1

		fig = plt.figure(1)
		fig.add_subplot(1,2,1)
		plt.imshow(image)
		# plt.show()
		# show prediction image with superpixels
		# plt.imshow(mark_boundaries(image, feature.superpixel))
		# plt.show()
		fig.add_subplot(1,2,2)
		plt.imshow(diff)
		plt.show()


if __name__ == '__main__':
	road_est = road_estimation("bag")
	road_est.train()
	road_est.test()
	# road_est.showPredictionImage()
