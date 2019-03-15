#!/usr/bin/python
#coding=UTF-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
	
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import json
from PIL import Image
from glob import glob
# import cv2
# import shutil
# from sklearn.cluster import MeanShift

from facenet import facenet
from facenet.align import detect_face

model = '20180402-114759'

sqrt_threshold = 1.05

def compare(imageList):
	images, face_area, file_list = load_and_align_data(imageList, 160, 44, 1.0)
	# images = images[0:100]
	with tf.Graph().as_default():
		with tf.Session() as sess:
			# Load the model
			facenet.load_model(model)

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			# Run forward pass to calculate embeddings
			feed_dict = { images_placeholder: images, phase_train_placeholder: False }
			emb = sess.run(embeddings, feed_dict=feed_dict)
	return emb, file_list, face_area
	# return emb[0]
			# nrof_images = len(face_area)

			# print('Images:')
			# for i in range(nrof_images):
			# 	print('%1d: %s' % (i, imageList[i]))
			# print('')
			
			# Print distance matrix
			# print('Distance matrix')
			# print('	', end='')
			# for i in range(nrof_images):
			# 	print('	%1d	 ' % i, end='')
			# print('')
			# for i in range(nrof_images):
				# face_area[i].save('%s.png'%i)
				# cv2.imwrite('%s.png'%i, face_area[i], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
				# print('%1d  ' % i, end='')
				# for j in range(nrof_images):
				# 	dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
				# 	print i, j, dist
					# print('  %1.4f  ' % dist, end='')
				# print('')
	# print emb.shape
	# ms = MeanShift(sqrt_threshold)
	# ms.fit(emb)
	# ##每个点的标签
	# labels = ms.labels_
	# print(labels)
	# for idx in xrange(len(labels)):
	# 	labelDir = 'classes/%s'%labels[idx]
	# 	if not os.path.isdir(labelDir):
	# 		os.makedirs(labelDir)
	# 	face_area[idx].save('%s/%s.png'%(labelDir, idx))
	##簇中心的点的集合
	# cluster_centers = ms.cluster_centers_
	# ##总共的标签分类
	# labels_unique = np.unique(labels)
	# ##聚簇的个数，即分类的个数
	# n_clusters_ = len(labels_unique)

	# print("number of estimated clusters : %d" % n_clusters_)

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	
	print('Creating networks and loading parameters')
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

	tmp_image_paths = copy.copy(image_paths)
	img_list = []
	face_area = []
	file_list = []
	for image in tmp_image_paths:
		img = misc.imread(os.path.expanduser(image), mode='RGB')
		img_size = np.asarray(img.shape)[0:2]
		bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
		if len(bounding_boxes) < 1:
			image_paths.remove(image)
			print("can't detect face, remove ", image)
			continue
		for box in bounding_boxes:
			det = np.squeeze(box[0:4])
			bb = np.zeros(4, dtype=np.int32)
			bb[0] = np.maximum(det[0]-margin/2, 0)
			bb[1] = np.maximum(det[1]-margin/2, 0)
			bb[2] = np.minimum(det[2]+margin/2, img_size[1])
			bb[3] = np.minimum(det[3]+margin/2, img_size[0])
			cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
			aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			img_list.append(prewhitened)
			face_area.append(Image.open(image).crop((bb[0], bb[1], bb[2], bb[3])))
			file_list.append(image)
	images = np.stack(img_list)
	return images, face_area, file_list

if __name__ == '__main__':
	# main(parse_arguments(sys.argv[1:]))
	imageList = glob('frame/*.jpg')
	# imageList = ["frame/041.jpg"]
	imageList.sort()
	labelList = glob('label/*.jpg')
	labelList.sort()
	labelData, _ = compare(labelList)
	print len(labelList), len(labelData)
	step = 50
	resultDict = {}
	for i in xrange(0, len(imageList), step):
		result, file_list, face_area = compare(imageList[i:i+step])
		for j in xrange(len(labelData)):
			label = os.path.basename(labelList[j]).split('.')[0]
			for k in xrange(len(result)):
				dist = np.sqrt(np.sum(np.square(np.subtract(labelData[j], result[k]))))
				if dist <= sqrt_threshold:
					resultDict.setdefault(label, [])
					# file_list[k].save('%s.jpg'%k)
					resultDict[label].append(file_list[k])
					print label, file_list[k], dist
	print json.dumps(resultDict)