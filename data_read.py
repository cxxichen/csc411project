import scipy.io
# load data from files
def data_load():
	train_data = scipy.io.loadmat("train_data.mat")
	test_data = scipy.io.loadmat("test_data.mat")
	valid_data = scipy.io.loadmat("valid_data.mat")
	return train_data["train_data"], train_data["train_labels"].ravel(), valid_data["valid_data"], \
	valid_data["valid_labels"].ravel(), test_data["test_data"], test_data["test_label"].ravel()

def load_valid_file():
	train_data = scipy.io.loadmat("train_data.mat")
	test_data = scipy.io.loadmat("test_data.mat")
	valid_data = scipy.io.loadmat("valid_data.mat")
	train_matrices = scipy.io.loadmat("train_matrices")

	valid_files_count = valid_data["valid_files_count"]
	valid_files = valid_data["valid_files"]
	valid_superpixels = valid_data["valid_superpixels"]
	valid_pixels_labels = train_matrices["valid_pixels_labels"]
	return valid_files_count, valid_files, valid_superpixels, valid_pixels_labels

def load_test_file():
	train_data = scipy.io.loadmat("train_data.mat")
	test_data = scipy.io.loadmat("test_data.mat")
	valid_data = scipy.io.loadmat("valid_data.mat")
	train_matrices = scipy.io.loadmat("train_matrices")

	test_files_count = test_data["test_files_count"]
	test_files = test_data["test_files"]
	test_superpixels = test_data["test_superpixels"]
	test_pixels_labels = train_matrices["test_pixels_labels"]
	return test_files_count, test_files, test_superpixels, test_pixels_labels
