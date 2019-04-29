#!/usr/bin/env python3
# ok

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as utils
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import os
import sys
import math

import json
import cv2

import shutil

class Data():
	def __init__(self, filenames, path='./data/seaturtle.coco/images/train2019/', train=True):
		self.data = filenames
		self.train=train

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		

		angle = int(np.random.uniform(0,360,1))

		image = transforms.ToPILImage()(image)
		image = transforms.functional.affine(image,angle,(0,0),1,0)
		image = transforms.ToTensor()(image)
		image = transforms.Normalize((0.1307,), (0.3081,))(image)

		return image, angle

def get_filenames(json_file):
	annotations = json.load(open(json_file,'r'))
	print(len(annotations['annotations']))
	print(len(annotations['images']))

	bbox = next((annotation['bbox'] for annotation in annotations['annotations'] if annotation['id'] == 28), None)
	filename = next((image['file_name'] for image in annotations['images'] if image['id'] == 28), None)

	print(filename, bbox)
	bbox = [int(x) for x in bbox]
	print('./data/seaturtle.coco/images/val2019/'+filename)
	im = cv2.imread('./data/coco/seaturtle.coco/images/val2019/'+filename)
	cv2.rectangle(im,tuple(bbox[:2]),tuple(bbox[2:]),color=(0,255,0),thickness=3)
	cv2.imshow('im',im)
	cv2.waitKey(0)

	exit(1)

	return [filename]

def main():
	#print('Loading Dataset...')

	path = './data/coco/seaturtle.coco/annotations/instances_%s2019.json'%('val')
	filenames = get_filenames(path)

	dataset = Data(filenames=filenames,
						 train=True)
	datasetLoader = DataLoader(dataset,batch_size=6,shuffle=True)
	dataset_iter = iter(datasetLoader)

	#print('Successfully Loaded Dataset')

	images, labels = dataset_iter.next()
	grid = torchvision.utils.make_grid(images)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	plt.axis('off')
	plt.title(labels.numpy());
	plt.show()	

	
if __name__=='__main__':
	main()
