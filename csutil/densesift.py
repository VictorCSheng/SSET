import numpy as np
import cv2

class DenseSIFT(object):
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=1)

	def detectAndCompute(self, image, step_size=12, window_size=(10, 10)): # 12 10
		if window_size is None:
			winH, winW = image.shape[:2]
			window_size = (winW // 4, winH // 4)

		if image.ndim == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		descriptors = np.array([], dtype=np.float32).reshape(0, 128)
		keypoints = []
		for cropinfo in self._crop_image(image, step_size, window_size):
			# crops_x, crops_y, crop = cropinfo
			crop_x = cropinfo[0]
			crop_y = cropinfo[1]
			crop = cropinfo[2]
			tmp_keypoints, tmp_descriptor = self.sift.detectAndCompute(crop, None)
			# tmp_keypoints, tmp_descriptor = self._detectAndCompute(crop)
			# tmp_descriptor = self._detectAndCompute(crop)[1]
			if tmp_descriptor is None:
				continue
			descriptors = np.vstack([descriptors, tmp_descriptor])

			for i in range(0,len(tmp_keypoints)):
				tmp_keypoints[i].pt = [tmp_keypoints[i].pt[0] + crop_y, tmp_keypoints[i].pt[1] + crop_x]

			keypoints = keypoints + tmp_keypoints
			# keypoints.append(tmp_keypoints)
			# keypoints = np.vstack([keypoints, tmp_keypoints])
		return keypoints, descriptors

	# def _detect(self, image):
	# 	return self.sift.detect(image)
	#
	# def _compute(self, image, kps, eps=1e-7):
	# 	kps, descs = self.sift.compute(image, kps)
	#
	# 	if len(kps) == 0:
	# 		return [], None
	#
	# 	descs /= (descs.sum(axis=1, keepdims=True) + eps)
	# 	descs = np.sqrt(descs)
	# 	return kps, descs
	#
	# def _detectAndCompute(self, image):
	# 	kps = self._detect(image)
	# 	return self._compute(image, kps)

	def _sliding_window(self, image, step_size, window_size):
		for y in range(0, image.shape[0], step_size):
			for x in range(0, image.shape[1], step_size):
				yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

	def _crop_image(self, image, step_size, window_size):
		# crops = []
		# crops_x = []
		# crops_y = []
		crops_info= []
		winH, winW = window_size
		for (x, y, window) in self._sliding_window(image, step_size=step_size, window_size=(winW, winH)):
			if window.shape[0] != winH or window.shape[1] != winW:
				continue   # 最后不够一个了就跳过

			crops_infotemp =[x,y,np.array(window)]
			crops_info.append(crops_infotemp)
			# crops.append(window)
			# crops_x.append(x)
			# crops_y.append(y)

		return crops_info