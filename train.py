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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batchSz', type=int, default=60)
	parser.add_argument('--nEpochs', type=int, default=20)
	parser.add_argument('--no-cuda', action='store_true')
	parser.add_argument('--save')
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--opt', type=str, default='adam', 
							choices=('sgd', 'adam', 'rmsprop'))
	parser.add_argument('--no-resume', action='store_true')
	parser.add_argument('--example',action='store_true')
	parser.add_argument('--type',default='regression')
	parser.add_argument('--test',action='store_true')
	parser.add_argument('--nClasses', type=int, default=None)
	parser.add_argument('--data-type', type=str, default='turtles', 
							choices=('turtles', 'mnist'))
	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	# create save directory for state dict, labeled with algorithm type and data type
	args.save = args.save or 'work/densenet.%s.%s'%(args.data_type,args.type)

	if(args.no_resume and os.path.exists(args.save)):
		i = input("Warning: Deleting Save Directory. Continue? (y/n) ")
		if(i!='y'):
			print("Program Cancelled")
			exit(1)	



	if(args.nClasses == None):
		if(args.type.startswith('classification')):
			args.nClasses = 360
		elif(args.type.startswith('regression')):
			args.nClasses = 1
		else:
			print("Invalid type provided, see -h for help")
			exit(1)


	if args.example:
		print("Showing Example of Weights")
		if(not os.path.exists(args.save)):
			print("Save path does not exist")
			exit(1)
		net = DenseNet(growthRate=12, depth=100, compression=0.5,
							bottleneck=True, args=args)
		print('Loading Saved Parameters...')
		if(args.cuda):
			net.load_state_dict(torch.load(os.path.join(args.save,'latest-%s.pth'%(args.type))))
		else:
			net.load_state_dict(torch.load(os.path.join(args.save,'latest-%s.pth'%(args.type)),map_location=lambda storage, location: 'cpu'))

		if(args.data_type=='turtles'):
			test_dataset = Data_turtles(dataType='test2019',experiment_type='example', args = args)
		if(args.data_type=='mnist'):
			test_dataset = Data_mnist(path='./data/mnist/mnist_test.csv',
							experiment_type='example',args=args)
		if(args.batchSz>8):
			args.batchSz=8
		testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=True)
		test_iter = iter(testLoader)
		print('Successfully Loaded Parameters')

		# net.eval()
		images_normalized, images, angles, labels = next(test_iter)

		# if args.cuda:
		# 		images_normalized, angles = images_normalized.cuda(), angles.cuda()
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

		return

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	if not args.no_resume:
		if(not os.path.exists(args.save)):
			print("Save path does not exist")
			exit(1)

		net = DenseNet(growthRate=12, depth=100, compression=0.5,
							bottleneck=True, args=args)
		print('Loading Saved Parameters...')
		net.load_state_dict(torch.load(os.path.join(args.save,'latest-%s.pth'%(args.type))))
		trainF = open(os.path.join(args.save, 'train.csv'), 'a')
		testF = open(os.path.join(args.save, 'test.csv'), 'a')
		print('Successfully Loaded Parameters')
	else:
		net = DenseNet(growthRate=12, depth=100, compression=0.5,
							bottleneck=True, args=args)
		# if(os.path.isfile('work/pretrain/densenet121-a639ec97.pth')):
		# 	net.load_state_dict(torch.load('work/pretrain/densenet121-a639ec97.pth'))
		# else:
		# 	print("no pretrain found")
		if os.path.exists(args.save):
			shutil.rmtree(args.save)
		os.makedirs(args.save, exist_ok=True)
		trainF = open(os.path.join(args.save, 'train.csv'), 'w')
		testF = open(os.path.join(args.save, 'test.csv'), 'w')

	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	

	# laod training set
	if(args.data_type=='turtles'):
		train_dataset = Data_turtles(dataType = 'train2019', experiment_type='train', args = args)
	if(args.data_type=='mnist'):
		train_dataset = Data_mnist(path='./data/mnist/mnist_train.csv',
						experiment_type='train',args=args)

	# laod validation set
	if(args.data_type=='turtles'):
		val_dataset = Data_turtles(dataType = 'val2019', experiment_type='validation', args = args)
	if(args.data_type=='mnist'):
		val_dataset = Data_mnist(path='./data/mnist/mnist_val.csv',
						experiment_type='validation',args=args)

	# load test set
	if(args.data_type=='turtles'):
		test_dataset = Data_turtles(dataType='test2019', experiment_type='test', args = args)
	if(args.data_type=='mnist'):
		test_dataset = Data_mnist(path='./data/mnist/mnist_test.csv',
						experiment_type='test',args=args)

	trainLoader = DataLoader(train_dataset,batch_size=args.batchSz,shuffle=True)
	train_iter = iter(trainLoader)
	
	valLoader = DataLoader(val_dataset,batch_size=args.batchSz,shuffle=True)
	val_iter = iter(trainLoader)

	testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=False, drop_last=True)
	test_iter = iter(testLoader)
	print('Successfully Loaded Dataset')

	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))
	if args.cuda:
		net = net.cuda()
	if args.opt == 'sgd':
		optimizer = optim.SGD(net.parameters(), lr=1e-1,
							momentum=0.9, weight_decay=1e-4)
	elif args.opt == 'adam':
		optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
	elif args.opt == 'rmsprop':
		optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)
	

	for epoch in range(1, args.nEpochs + 1):
		adjust_opt(args.opt, optimizer, epoch)
		if(not args.test):
			train(args, epoch, net, trainLoader, optimizer, trainF)
			torch.save(net.state_dict(), os.path.join(args.save, 'latest-%s.pth'%(args.type)))
		
		print()
		if(args.test):
			test(args, epoch, net, testLoader, optimizer, testF)
		# os.system('./plot.py {} &'.format(args.save))
		if(args.test):
			return

	trainF.close()
	testF.close()

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

def train(args, epoch, net, trainLoader, optimizer, trainF):
	net.train()
	nProcessed = 0
	nTrain = len(trainLoader.dataset)
	for batch_idx, (data, target) in enumerate(trainLoader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = net(data)

		for i in range(args.batchSz):
			print(output.data[i],target.data[i])

		if(args.type.startswith('classification')):
			loss = F.nll_loss(output, target)
		if(args.type.startswith('regression')):
			loss = mse(output.float(), target.float())

		

		loss.backward()
		optimizer.step()
		nProcessed += len(data)
		if(args.type.startswith('classification')):
			pred = output.data.max(1)[1] # get the index of the max log-probability
			incorrect = pred.ne(target.data).cpu().sum()
			err = 100.*incorrect/len(data)
		if(args.type.startswith('regression')):
			err = torch.mean(abs(output.float().squeeze() - target.float()))

		partialEpoch = epoch + batch_idx / len(trainLoader) - 1
		print('\r',end='')
		print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
			partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
			loss.data[0], err),end='')


		trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
		trainF.flush()	

def test(args, epoch, net, testLoader, optimizer, testF):
	# net.eval()
	test_loss = 0
	incorrect = 0
	all_pred = np.array([])
	all_targ = np.array([])
	all_labl = np.array([])
	all_diff = np.array([])
	for data, display_image, target, label in testLoader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		output = net(data)
		# for i in range(args.batchSz):
		# 	print(output.data[i],target.data[i])
		if(args.type.startswith('classification')):
			test_loss += F.nll_loss(output, target).data[0]
			pred = output.data.max(1)[1] # get the index of the max log-probability
			incorrect += pred.ne(target.data).cpu().sum()
		if(args.type.startswith('regression')):
			test_loss += mse(output, target.float()).data
			pred = output.data.squeeze()
			incorrect = 0

		


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

	testF.write('{},{},{}\n'.format(epoch, test_loss, err))
	testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
	if optAlg == 'sgd':
		if epoch < 150: lr = 1e-1
		elif epoch == 150: lr = 1e-2
		elif epoch == 225: lr = 1e-3
		else: return

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

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
	def __init__(self, growthRate=12, depth=100, compression=0.5, bottleneck=True, args=None):
		super(DenseNet, self).__init__()
		
		self.args = args

		# split up layers between three dense blocks 
		# other than initial conv, two transition, and terminating fc
		nDenseBlocks = (depth-4) // 3

		# bottleneck class 
		if bottleneck:
			nDenseBlocks //= 2


		# initial convolution to transform into desired input size
		nChannels = 2*growthRate
		if(self.args.data_type=='turtles'):
			self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
		if(self.args.data_type=='mnist'):
			self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1, bias=False)
		
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
		self.fc = nn.Linear(nChannels, args.nClasses)

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
		if(self.args.type.startswith('classification')):
			out = F.log_softmax(self.fc(out),dim=0)
		if(self.args.type.startswith('regression')):
			out = self.fc(out)
		return out

if __name__=='__main__':
	main()





