import pandas as pd
import numpy as np
import torchvision.transforms as transforms


def get_angle(args):
	if(args.type=='regression45'):
		angle = int(np.random.uniform(-45,45,1))
		if(angle<0):
			angle=360+angle
	else:
		angle = int(np.random.uniform(0,360,1))
	return angle

class Data_mnist():
	def __init__(self, path, experiment_type='',args=None):
		self.data = pd.read_csv(path)
		self.experiment_type=experiment_type
		self.args=args

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28,28,1))
		angle = get_angle(self.args)
		label = self.data.iloc[index, 0]

		image = transforms.ToPILImage()(image)
		image = transforms.functional.affine(image,angle,(0,0),1,0)
		image = transforms.ToTensor()(image)
		image_normalized = transforms.Normalize((0.1307,), (0.3081,))(image)
		
		if(self.args.type.startswith('classification')):
			# nCalsses should be a factor of 360
			angle = int(angle/int(360/self.args.nClasses))

		if(self.experiment_type=='example'):
			return image_normalized, image, angle
		if(self.experiment_type=='test'):
			return image_normalized, image, angle, label

		# during training 
		return image_normalized, angle


import torch
from utils.coco import COCO
from utils.progress_bar.bar import Bar
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import pickle
from utils.bbox_util import *
import cv2
import os
plt.switch_backend('tkagg')

class Data_turtles():
	def __init__(self, dataDir = 'data/coco/seaturtle.coco', dataType='train2019',
					experiment_type = '', args = None):
		self.dataDir = dataDir
		self.dataType = dataType
		self.annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
		self.coco=COCO(self.annFile)
		# display COCO categories and supercategories
		cats = self.coco.loadCats(self.coco.getCatIds())
		nms=[cat['name'] for cat in cats]

		# print('\tCOCO categories: \n\t{}'.format(' '.join(nms)))
		# set category as supercategory to get all categories
		# nms = set([cat['supercategory'] for cat in cats])
		# print('\tCOCO supercategories: \n\t{}\n'.format(' '.join(nms)))

		# get all images containing given categories, select one at random
		self.catIds = self.coco.getCatIds(catNms=nms[1]);
		self.imgIds = self.coco.getImgIds(catIds=self.catIds);
		self.args = args
		self.experiment_type = experiment_type
		print("Loading Data: {}".format(dataType))
		self.preprocess_images()

		# need to shuffle if using pickle to load
		np.random.shuffle(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# preprocess all images
		image, poly, theta, view = self.data[index]
		# process as each image is loaded
		# image, poly, theta, view = self.get_image(index)

		resize_to = 32
		angle = get_angle(self.args)
		
		show=False
		
		theta = theta*180/np.pi
		h,w = image.shape[:2]
		cx,cy = w/2,h/2
		frame = np.array([[0,0],
						  [w,0],
						  [w,h],
						  [0,h]])
		
		

		mer = self.MER(poly,*image.shape[:2])
		I = rotate_im(image,theta)
		# poly_rot = rotate_box(poly,theta,cx,cy,h,w).reshape((4,2))
		# frame_rot = np.round(rotate_box(frame,theta,cx,cy,h,w).reshape((4,2)))
		mer_rot = np.round(rotate_box(mer,theta,cx,cy,h,w).reshape((4,2)))
		mir = self.MIR(theta*np.pi/180, mer[2,1]-mer[0,1], mer[1,0]-mer[0,0], self.MER(mer_rot, *I.shape[:2]))
		# mir = self.MIR(theta*np.pi/180, h, w, self.MER(mer_rot, *I_rot.shape[:2]))
		I = I[mir[0,1]:mir[2,1],mir[0,0]:mir[2,0]]

		# show the stages of cropping the image
		if(show):
			# =============================================
			# original image
			self.ax = plt.gca()
			self.show_annotation(poly)
			self.show_MER(mer)
			plt.imshow(image)
			plt.show()
			# =============================================
			# rotated image
			I_rot = rotate_im(image,theta)
			self.ax = plt.gca()
			self.show_MER(mer_rot)
			self.show_annotation(mir)
			plt.imshow(I_rot)
			plt.show()
			# =============================================
			# final cropped image
			I_cropped = I_rot[mir[0,1]:mir[2,1],mir[0,0]:mir[2,0]]
			plt.imshow(I_cropped)
			plt.show()
			# ==============================================

		I = transforms.ToPILImage()(I)
		I = transforms.Resize((resize_to,resize_to))(I)
		I = transforms.functional.affine(I,angle,(0,0),1,0)
		I = transforms.ToTensor()(I)
		image_normalized = transforms.Normalize(self.means, self.stds)(I)
		
		transform = transforms.Compose([transforms.ToPILImage(), 
									transforms.Resize((resize_to,resize_to)), 
									transforms.ToTensor()])
		image = transform(image)

		if(self.args.type.startswith('classification')):
			# nCalsses should be a factor of 360
			angle = int(angle/int(360/self.args.nClasses))

		if(self.experiment_type=='example'):
			return image_normalized, image, angle
		if(self.experiment_type=='test'):
			return image_normalized, image, angle, view

		# during training 
		return image_normalized, angle

	def MIR(self, angle, im_h, im_w, mer): 
		
		new_height = mer[2,1]-mer[0,1]
		new_width = mer[1,0]-mer[0,0]

		# ======================================
		"""
		Given a rectangle of size wxh that has been rotated by 'angle' (in
		radians), computes the width and height of the largest possible
		axis-aligned rectangle within the rotated rectangle.
		Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
		"""

		quadrant = int(np.floor(angle / (np.pi / 2))) & 3
		sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
		alpha = (sign_alpha % np.pi + np.pi) % np.pi

		bb_w = im_w * np.cos(alpha) + im_h * np.sin(alpha)
		bb_h = im_w * np.sin(alpha) + im_h * np.cos(alpha)

		gamma = np.arctan2(bb_w, bb_w) if (im_w < im_h) else np.arctan2(bb_w, bb_w)

		delta = np.pi - alpha - gamma

		length = im_h if (im_w < im_h) else im_w

		d = length * np.cos(alpha)
		a = d * np.sin(alpha) / np.sin(delta)

		y = a * np.cos(gamma)
		x = y * np.tan(gamma)

		rot_w = bb_w - 2 * x
		rot_h = bb_h - 2 * y
		# ======================================

		# Top-left corner
		top = y + mer[0,1]
		left= x + mer[0,0]

		# Bottom-right corner
		bottom = np.round(top + rot_h)
		right = np.round(left + rot_w)

		return np.array([[left,top],
						[right,top],
						[right,bottom],
						[left,bottom]]).astype(np.int32)	

	def MER(self, poly, im_h, im_w):
		left = np.min(poly[:,0])
		right = np.max(poly[:,0])
		top = np.min(poly[:,1])
		bottom = np.max(poly[:,1])

		left = max(left,0)
		right = min(right, im_w)
		top = max(top, 0)
		bottom = min(bottom, im_h)

		return np.array([[left,top],
						[right,top],
						[right,bottom],
						[left,bottom]])

	def show_MER(self, enclosing):
		polygons = [Polygon(enclosing)]
		p = PatchCollection(polygons, facecolor='none', edgecolors=(1,0,0), linewidths=2)
		self.ax.add_collection(p)

	# returns the cropped image and ratio of new h/w to old h/w
	def crop_to_poly(self, poly, im_h, im_w, im_tensor):
		left = round(poly[0,0])
		right = round(poly[1,0])
		top = round(poly[0,1])
		bottom = round(poly[2,1])


		return im_tensor[top:bottom,left:right,:]

	def clip_poly(self,poly, im_h, im_w):

		poly[:,0] = np.where(poly[:,0]>im_w,im_w,poly[:,0])
		poly[:,0] = np.where(poly[:,0]<0,0,poly[:,0])
		poly[:,1] = np.where(poly[:,1]>im_h,im_h,poly[:,1])
		poly[:,1] = np.where(poly[:,1]<0,0,poly[:,1])

		return poly

	def show_annotation(self, poly):
		color = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
		polygons = [Polygon(poly)]
		# inside shading
		p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
		self.ax.add_collection(p)
		# bordering lines
		p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
		self.ax.add_collection(p)

	def test_image(self, image, poly, theta, view):
		resize_to = 32
		angle = get_angle(self.args)
		theta = theta*180/np.pi
		h,w = image.shape[:2]
		cx,cy = w/2,h/2
		mer = self.MER(poly,*image.shape[:2])
		I = rotate_im(image,theta)
		mer_rot = np.round(rotate_box(mer,theta,cx,cy,h,w).reshape((4,2)))
		mir = self.MIR(theta*np.pi/180, mer[2,1]-mer[0,1], mer[1,0]-mer[0,0], self.MER(mer_rot, *I.shape[:2]))
		I = I[mir[0,1]:mir[2,1],mir[0,0]:mir[2,0]]
		I = transforms.ToPILImage()(I)
		I = transforms.Resize((resize_to,resize_to))(I)
		I = transforms.functional.affine(I,angle,(0,0),1,0)
		I = transforms.ToTensor()(I)
		image_normalized = transforms.Normalize((1,1,1), (.5,.5,.5))(I)
		return True


	def get_image(self, index):
		ID = self.imgIds[index]
		img = self.coco.loadImgs(ID)[0]
		I = io.imread('%s/images/%s/%s'%(self.dataDir,self.dataType,img['file_name']))
		
		annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
		try:
			turtle_body, turtle_head = self.coco.loadAnns(annIds)
		except:
			# print(self.coco.loadAnns(annIds))
			print(len(self.coco.loadAnns(annIds)))
			print("Error loading image")
			exit(1)

		return (I,turtle_head['segmentation'],turtle_head['theta'],turtle_head['viewpoint'])
		

	def preprocess_images(self):

		# when loading one image at a time
		# if(os.path.isfile("data/loaded_data_{}_means_stds.p".format(self.dataType))):
		# 	print("Pickle file found, loading means and stdevs...")
		# 	(self.means,self.stds) = pickle.load(open("data/loaded_data_{}_means_stds.p".format(self.dataType), "rb" ))
		# 	return 

		# when loading whole dataset
		if(os.path.isfile("data/loaded_data_{}.p".format(self.dataType))):
			print("Pickle file found, loading data...")
			(self.data, self.means,self.stds) = pickle.load(open("data/loaded_data_{}.p".format(self.dataType), "rb" ))
			print(self.dataType,len(self.data))
			return 

		bar = Bar("Preprocessing",max=len(self.imgIds))
		self.data = []

		N = len(self.imgIds)
		means = [0,0,0]
		stds = [0,0,0]
		np.random.shuffle(self.imgIds)
		for i,ID in enumerate(self.imgIds):
			

			img = self.coco.loadImgs(ID)[0]
			I = io.imread('%s/images/%s/%s'%(self.dataDir,self.dataType,img['file_name']))
			# plt.axis('off')

			# for each channel, sum each mean and variance divided by the dataset size
			for channel in range(3):
				means[channel] += np.mean(I[:,:,channel])/N
				stds[channel] += np.std(I[:,:,channel])**2/N


			# plt.imshow(I)
			annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
			try:
				turtle_body, turtle_head = self.coco.loadAnns(annIds)
			except:
				# print(self.coco.loadAnns(annIds))
				print("\nmultiple ann")
				print(len(self.coco.loadAnns(annIds)))
				continue

			poly = turtle_head['segmentation']
			poly = np.array(poly).reshape((5,2))
			poly = poly[:-1,:]

			# test to see if augmentation works on image
			# for filtering out pooly annotated images
			try:
				self.test_image(I,poly,turtle_head['theta'],turtle_head['viewpoint'])
			except:
				print("\nskipped",ID)
				continue

			# h,w = I.shape[:2]
			# if(min(np.min(poly),0)<0 or max(np.max(poly[:,0]),w)>w or max(np.max(poly[:,1]),h)>h):
			# 	continue

			self.data.append((I,poly,turtle_head['theta'],turtle_head['viewpoint']))
			bar.next()

		stds = np.sqrt(stds)
		print()

		self.means = means
		self.stds = stds

		# have the option of loading whole dataset or just means and stdevs
		pickle.dump((self.data,means,stds), open("data/loaded_data_{}.p".format(self.dataType), "wb" ))
		pickle.dump((means,stds), open("data/loaded_data_{}_means_stds.p".format(self.dataType), "wb" ))


	
		
		

	