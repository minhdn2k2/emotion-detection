import numpy as np
# from keras.utils import Sequence
import cv2
from keras_vggface import utils

class predict_data_sequence:
		def __init__(self, features, target_dim=(224,224)):
				self.features = features
				self.target_dim = target_dim
				self.target_channels = 3

		def __gray2RGB__(self, x):
			if len(x.shape)==2:
				return np.stack((x,x,x),-1)
			else:
				assert len(x.shape)==3
				if len(x[0,0,:]) == 1:
					return np.stack((x[:,:,0],x[:,:,0],x[:,:,0]),-1)
				else:
					assert len(x[0,0,:])==self.target_channels
			return x

		def get_data(self):
				X = []
				for feature in self.features:   
						x = feature
						x = cv2.resize(x,self.target_dim,interpolation=cv2.INTER_CUBIC)
						x = self.__gray2RGB__(x)
						x = utils.preprocess_input(x, version=2)

						X.append(x)

				X = np.array(X)
				return X
	
def gray_to_rgb(x):
	if len(x.shape) == 2:
		return np.stack((x, x, x), -1)
	else:
		assert len(x.shape) == 3
		if len(x[0,0,:]) == 1:
			return np.stack((x[:, :, 0], x[:, :, 0],x[:, :, 0]), -1)
		else:
			assert len(x[0, 0, :]) == 3
	return x


def transfer_data(imgs, target_dim=(224,224)):
	X = []
	for img in imgs:
		x = img

		x = cv2.resize(x, target_dim, interpolation=cv2.INTER_CUBIC)

		x = gray_to_rgb(x)
		x = utils.preprocess_input(x, version=2)
		X.append(x)

	X = np.array(X)
	return X

def transfer_data_by_crop(points, img, target_dim=(224,224)):
	X = []

	for point in points:
		bbox = point['bbox']
		x = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].astype(np.float64)

		if x.shape[0] != 0 and x.shape[1] != 0:
			x = cv2.resize(x, target_dim, interpolation=cv2.INTER_CUBIC)
			x = gray_to_rgb(x)
			# x = utils.preprocess_input(x, version=2)
			X.append(x)
	X = np.array(X)

	return X




	