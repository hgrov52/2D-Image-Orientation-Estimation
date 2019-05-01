from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')

from data import *

# =====================================================

def handle_args(args):
	handle_example(args)
	handle_nClasses(args)

def handle_example(args):
	n,m,testLoader = get_data_loaders(args)

def handle_nClasses(args):	
	if(args.nClasses==None):
		if(args.type.endswith('8')):
			args.nClasses = 8
		elif(args.type.startswith('classification')):
			args.nClasses = 360
		else:
			args.nClasses = 1

def get_data_loaders(args):
	if(args.data_type=='turtles'):
		train_dataset = Data_turtles(dataType = 'train2019', experiment_type='train', args = args)
		val_dataset = Data_turtles(dataType = 'val2019', experiment_type='validation', args = args)
		test_dataset = Data_turtles(dataType='test2019', experiment_type='test', args = args)

	if(args.data_type=='mnist'):
		train_dataset = Data_mnist(path='./data/mnist/mnist_train.csv',
						experiment_type='train',args=args)
		val_dataset = Data_mnist(path='./data/mnist/mnist_val.csv',
						experiment_type='validation',args=args)		
		test_dataset = Data_mnist(path='./data/mnist/mnist_test.csv',
						experiment_type='test',args=args)		

	trainLoader = DataLoader(train_dataset,batch_size=args.batchSz,shuffle=True)
	valLoader = DataLoader(val_dataset,batch_size=args.batchSz,shuffle=True)
	testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=False, drop_last=True)
	
	print('Successfully Loaded Dataset')

	return trainLoader, valLoader, testLoader

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

	handle_args(args)

	trainLoader, valLoader, testLoader = get_data_loaders(args)

	model = models.densenet161(pretrained=True)
	model.classifier = nn.Sequential(
                      nn.Linear(3, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, args.nClasses),                   
                      nn.LogSoftmax(dim=1))

	print(model.classifier)

if __name__=='__main__':
	main()