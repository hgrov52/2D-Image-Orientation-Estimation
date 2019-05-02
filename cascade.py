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
import shutil

from utils.test import test_stats

# imort data loader for both mnist and turtles
from data import *

import matplotlib.pyplot as plt
plt.switch_backend('tkagg')

def get_data_loaders(args):
	test_dataset = Data_turtles(dataType='test2019', experiment_type='test', args = args)
	testLoader = DataLoader(test_dataset,batch_size=1,shuffle=False, drop_last=True)
	return testLoader

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batchSz', type=int, default=1)
	parser.add_argument('--save')
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--opt', type=str, default='adam', 
							choices=('sgd', 'adam', 'rmsprop'))
	parser.add_argument('--no-resume', action='store_true')
	parser.add_argument('--example',action='store_true')
	parser.add_argument('--type',default='regression')
	
	args = parser.parse_args()
	


	course_type = 'classification4'
	course_save = 'work/densenet.%s.%s'%('turtles',course_type)
	course_nClasses = 4

	fine_type = 'regression45'
	fine_save = 'work/densenet.%s.%s'%('turtles',fine_type)
	fine_nClasses = 1

	cuda = torch.cuda.is_available()

	args.type=fine_type
	args.nClasses=fine_type
	args.cuda=cuda
	testLoader = get_data_loaders(args)

	if(not os.path.exists(course_save)):
		print("Course save path does not exist")
		exit(1)

	if(not os.path.exists(fine_save)):
		print("Fine save path does not exist")
		exit(1)


	course_net = DenseNet(growthRate=12, depth=100, compression=0.5,
						bottleneck=True, nClasses=course_nClasses, _type=course_type)
	fine_net = DenseNet(growthRate=12, depth=100, compression=0.5, 
						bottleneck=True, nClasses=fine_nClasses, _type=fine_type)
	
	print('Loading Saved Parameters...')
	if(cuda):
		course_net.load_state_dict(torch.load(os.path.join(course_save,'latest-%s.pth'%(course_type))))
	else:
		course_net.load_state_dict(torch.load(os.path.join(course_save,'latest-%s.pth'%(course_type)),map_location=lambda storage, location: 'cpu'))

	if(cuda):
		fine_net.load_state_dict(torch.load(os.path.join(fine_save,'latest-%s.pth'%(fine_type))))
	else:
		fine_net.load_state_dict(torch.load(os.path.join(fine_save,'latest-%s.pth'%(fine_type)),map_location=lambda storage, location: 'cpu'))

	# Test
	# test(args, course_net, fine_net, testLoader)

	# Example

	testLoader = get_data_loaders(args)
	test_iter = iter(testLoader)
	images_normalized, images, angles, labels = next(test_iter)

	data, target = Variable(images_normalized), Variable(angles)
	output = net(data)
	if(args.type.startswith('classification')):
		pred = output.data.max(1)[1] # get the index of the max log-probability
		#pred *= 360//args.nClasses
		print(pred)
	if(args.type.startswith('regression')):
		pred = output.data.reshape((1,args.batchSz))[0]

	images = [transforms.ToPILImage()(image) for image in images]	
	images = [transforms.functional.affine(image,-angle,(0,0),1,0) for image,angle in zip(images,pred)]
	images = [transforms.ToTensor()(image) for image in images]	

	grid = utils.make_grid(images)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	plt.axis('off')
	plt.title(args.type+'\n'+str(angles.numpy())+'\n'+str(pred.int().numpy()))
	plt.show()  



def mse(t1, t2):
	diff = torch.abs(t1.squeeze()-t2.squeeze())
	torch.where(diff>180,360-diff,diff)
	m = torch.mean(diff)
	return m*m

def L1(t1, t2):
	diff = torch.abs(t1.squeeze()-t2.squeeze())
	torch.where(diff>180,360-diff,diff)
	return torch.mean(diff)

def difference(args,t1,t2):
	diff = abs(t1-t2)
	# if(args.type.endswith("45")):
	# 	diff = np.where(diff>90,180-diff,diff)
	# else:
	diff = np.where(diff>180,360-diff,diff)
	return diff

def test(args, course_net, fine_net, testLoader):
	# net.eval()
	test_loss = 0
	incorrect = 0
	all_pred = np.array([])
	all_targ = np.array([])
	all_labl = np.array([])
	all_diff = np.array([])
	for data, display_image, target, label in testLoader:
		data, target = Variable(data), Variable(target)
		output = course_net(data)
		pred = output.data.max(0)[1]
		pred *= 90

		data = data.squeeze()
		data = transforms.ToPILImage()(data)
		data = transforms.functional.affine(data,-pred,(0,0),1,0)
		data = transforms.ToTensor()(data)
		data = data.unsqueeze(0)

		output = fine_net(data)
		print(output,target)


		test_loss += mse(output, target.float()).data
		pred = output.data.squeeze()

		predn = pred.cpu().numpy()
		targn = target.cpu().numpy()

		all_labl = np.hstack((all_labl, label))
		all_pred = np.hstack((all_pred, predn))
		all_targ = np.hstack((all_targ, targn))
		
		all_diff = np.hstack((all_diff, difference(args,predn, targn)))
	
	all_pred = np.where(all_pred>360,all_pred%360,all_pred)

	test_stats(args, all_pred, all_targ, all_labl, all_diff)

	test_loss /= len(testLoader) # loss function already averages over batch size
	nTotal = len(testLoader.dataset)
	err = 100.*incorrect/nTotal
	print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
		test_loss, incorrect, nTotal, err))

# ==============================================
# ==============================================

class Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(Bottleneck, self).__init__()
		interChannels = 4*growthRate
		self.batchnorm1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
		self.batchnorm2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		# initial 1x1 conv
		out = self.conv1(F.relu(self.batchnorm1(x)))
		# second 3x3 conv 
		out = self.conv2(F.relu(self.batchnorm2(out)))
		# concatenates x with output form this conv layer
		out = torch.cat((x, out), 1)
		return out

# layer with no bottleneck
class SingleLayer(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(SingleLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		# one 3x3 conv layer
		out = self.conv1(F.relu(self.bn1(x)))
		out = torch.cat((x, out), 1)
		return out

class Transition(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = F.avg_pool2d(out, 2)
		return out


class DenseNet(nn.Module):
	def __init__(self, growthRate=12, depth=100, compression=0.5, bottleneck=True, nClasses=1, _type=None):
		super(DenseNet, self).__init__()
		
		self._type = _type

		# split up layers between three dense blocks 
		# other than initial conv, two transition, and terminating fc
		nDenseBlocks = (depth-4) // 3

		# bottleneck class 
		if bottleneck:
			nDenseBlocks //= 2


		# initial convolution to transform into desired input size
		nChannels = 2*growthRate
		self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)

		# dense block 1
		self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
		nChannels += nDenseBlocks*growthRate

		# transition layer 1
		nOutChannels = int(math.floor(nChannels*compression))
		self.trans1 = Transition(nChannels, nOutChannels)

		# dense block 2
		nChannels = nOutChannels
		self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
		nChannels += nDenseBlocks*growthRate

		# transition laye 2
		nOutChannels = int(math.floor(nChannels*compression))
		self.trans2 = Transition(nChannels, nOutChannels)

		# dense block 3
		nChannels = nOutChannels
		self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
		nChannels += nDenseBlocks*growthRate

		# final fully connected layer
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.fc = nn.Linear(nChannels, nClasses)

		# initializations
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()


	# connect all layers of the dense blocks
	def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
		layers = []
		for i in range(int(nDenseBlocks)):
			if bottleneck:
				layers.append(Bottleneck(nChannels, growthRate))
			else:
				layers.append(SingleLayer(nChannels, growthRate))
			nChannels += growthRate
		return nn.Sequential(*layers)

	# run the whole network
	def forward(self, x):
		out = self.conv1(x)
		out = self.trans1(self.dense1(out))
		out = self.trans2(self.dense2(out))
		out = self.dense3(out)
		out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 7))
		if(self._type.startswith('classification')):
			out = F.log_softmax(self.fc(out),dim=0)
		if(self._type.startswith('regression')):
			out = self.fc(out)
		return out

if __name__=='__main__':
	main()





