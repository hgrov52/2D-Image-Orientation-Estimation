# imports for showing turtles
from utils.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# imports for torch load jpg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys
import math
import json
import shutil
from utils.progress_bar.bar import Bar
# ======================================================
plt.switch_backend('tkagg')

class Data():
	def __init__(self, dataDir = 'data/coco/seaturtle.coco', dataType='val2019'):
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
		self.preprocess_images()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		show_original = True
		show_cropped = True

		I, poly, theta, view = self.data[index]

		poly = torch.tensor(poly)
		poly = poly.reshape((5,2))

		print(theta,'degrees', view)

		resize_to = 256
		cropped, annot = self.crop_to_MER(poly, *I.shape[:2], I)
		#poly = self.verify_poly_bounds(poly, *I.shape[:2])
		cropped = transforms.ToPILImage()(cropped)
		#cropped = transforms.Resize((resize_to,resize_to))(cropped)

		if(show_original):
			plt.axis('off')
			plt.title('original')
			I = transforms.ToPILImage()(I)
			plt.imshow(I)
			self.ax = plt.gca()
			self.show_annotation(poly)
			self.show_MER(poly, *I.size)
			plt.show()
		if(show_cropped):
			plt.axis('off')
			plt.title('cropped')
			plt.imshow(cropped)
			self.ax = plt.gca()
			# print(poly)
			# resize_to = float(resize_to)
			# other_poly = poly.clone()
			# other_poly[:,0] = poly[:,0].float()*ratio_w
			# other_poly[:,1] = poly[:,1].float()*ratio_h
			# print(other_poly)
			# poly = (other_poly.float()* (resize_to/poly.float())).long()
			# print(poly)
			self.show_annotation(annot)
			plt.show()
		cropped = transforms.ToTensor()(cropped)
		return cropped
		

	def preprocess_images(self):
		bar = Bar("Preprocessing",max=len(self.imgIds))
		self.data = []
		np.random.shuffle(self.imgIds)
		for i,ID in enumerate(self.imgIds):
			

			img = self.coco.loadImgs(ID)[0]
			I = io.imread('%s/images/%s/%s'%(self.dataDir,self.dataType,img['file_name']))
			# plt.axis('off')

			# plt.imshow(I)
			annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
			turtle_body, turtle_head = self.coco.loadAnns(annIds)
			# self.coco.showAnns(anns[1:])
			# print(anns[1:])
			# plt.show()

			self.data.append((I,turtle_head['segmentation'],turtle_head['theta'],turtle_head['viewpoint']))
			bar.next()

			#  ============

			if(i>10):
				break

		print()

	# make sure the cropped poly bounds are within the image frame
	def verify_poly_bounds(self, poly, im_h, im_w):
		z = torch.zeros(poly.shape)
		new_poly = torch.where(poly<0.0,z,poly.float())

		
		return new_poly


	def MER(self, poly, im_w, im_h):
		left = torch.min(poly[:,0])
		right = torch.max(poly[:,0])
		top = torch.min(poly[:,1])
		bottom = torch.max(poly[:,1])

		left = max(left,0)
		right = min(right, im_w)
		top = max(top, 0)
		bottom = min(bottom, im_h)

		enclosing = torch.tensor([[left,top],
							  [right,top],
							  [right,bottom],
							  [left,bottom]])
		return enclosing

	def show_MER(self, poly, im_w, im_h):
		enclosing = self.MER(poly, im_w, im_h)
		polygons = [Polygon(enclosing)]
		p = PatchCollection(polygons, facecolor='none', edgecolors=(1,0,0), linewidths=2)
		self.ax.add_collection(p)
			
	# returns the cropped image and ratio of new h/w to old h/w
	def crop_to_MER(self, poly, im_h, im_w, tensor):
		enclosing = self.MER(poly, im_w, im_h)
		left = enclosing[0,0]
		right = enclosing[1,0]
		top = enclosing[0,1]
		bottom = enclosing[2,1]

		annot = poly.clone()
		print(annot)
		#annot = torch.where(annot<0,torch.zeros(annot.shape),annot.float())
		#print(annot)
		annot[:,0] = poly[:,0] - max(torch.min(poly[:,0]),0)
		annot[:,1] = poly[:,1] - max(torch.min(poly[:,1]),0)
		

		return tensor[top:bottom,left:right,:],annot
		


	def show_annotation(self, poly):
		color = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
		polygons = [Polygon(poly)]
		# inside shading
		p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
		self.ax.add_collection(p)
		# bordering lines
		p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
		self.ax.add_collection(p)



def old_show_turtle():
	dataDir = 'data/coco/seaturtle.coco'
	dataType = 'val2019'
	annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
	coco=COCO(annFile)

	# display COCO categories and supercategories
	cats = coco.loadCats(coco.getCatIds())
	nms=[cat['name'] for cat in cats]

	# print('\tCOCO categories: \n\t{}'.format(' '.join(nms)))
	# set category as supercategory to get all categories
	# nms = set([cat['supercategory'] for cat in cats])
	# print('\tCOCO supercategories: \n\t{}\n'.format(' '.join(nms)))

	# get all images containing given categories, select one at random
	catIds = coco.getCatIds(catNms=nms[1]);
	imgIds = coco.getImgIds(catIds=catIds);
	img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
	I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
	plt.axis('off')

	plt.imshow(I)
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)
	coco.showAnns(anns[1:])

	plt.show()

def new_show_turtle():


	for i in range(1):
		dataset = Data()
		datasetLoader = DataLoader(dataset,batch_size=1,shuffle=True)
		datasetIter = iter(datasetLoader)
		batch = datasetIter.next()

		#I,poly = batch

		# load and display instance annotations
		# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
		# anns = coco.loadAnns(annIds)
		# turtle = anns[0]
		# turtle_head = anns[1]
		# head_bbox = [int(x) for x in turtle_head['bbox']]
		# head_seg_bbox = [int(x) for x in turtle_head['segmentation_bbox']]
		# seg = turtle_head['segmentation'][0]
		# poly = np.array(seg).reshape((int(len(seg)/2), 2))
		
		# try to eliminate duplicates
		# for i,point in enumerate(poly.copy()):
		# 	if(i>0 and i<poly.shape[0]-1 and point in poly[:i]+poly[i+1:]):
		# 		poly = poly[:i]+poly[i+1:]
		# poly = np.array(set(poly.tolist()))
		# print(poly.shape)

		# show points on image
		# for point in poly[:-1]:
		# 	cv2.circle(I, tuple(point),10,(0,255,0),-1)	
		# ========================

		# show_ann = True
		# print()
		# print(poly.shape)
		# enclosing_list = [MER(p, I.shape[0], I.shape[1]) for p in poly]
		# for poly, enclosing, image in zip(poly,enclosing_list,I):
		# 	if(show_ann):
		# 		show_annotation(poly)
		# 	plt.gca().add_collection(PatchCollection([Polygon(enclosing)], facecolor='none', edgecolors=(1,0,0), linewidths=2))
		# 	image = transforms.ToPILImage()(image)
		# 	plt.imshow(image)
		# 	plt.show()

if __name__ == '__main__':
	new_show_turtle()
