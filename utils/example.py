import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
import os
import torch

from .DenseNet import DenseNet
from .Data import Data

def example(args):
	print("Showing Example of Weights")
	if(not os.path.exists(args.save)):
		print("Save path does not exist")
		exit(1)
	net = DenseNet(growthRate=12, depth=100, reduction=0.5,
						bottleneck=True, nClasses=args.nClasses)
	print('Loading Saved Parameters...')
	if(args.cuda):
		net.load_state_dict(torch.load(os.path.join(args.save,'latest-%s.pth'%(args.type))))
	else:
		net.load_state_dict(torch.load(os.path.join(args.save,'latest-%s.pth'%(args.type)),map_location=lambda storage, location: 'cpu'))

	test_dataset = Data(path='./data/mnist/mnist_test.csv',
					 train=False,example=True)
	if(args.batchSz>8):
		args.batchSz=8
	testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=True)
	test_iter = iter(testLoader)
	print('Successfully Loaded Parameters')

	net.eval()
	images_normalized, images, labels = next(test_iter)

	# if args.cuda:
	# 		images_normalized, labels = images_normalized.cuda(), labels.cuda()
	data, target = Variable(images_normalized), Variable(labels)
	output = net(data)
	if(args.type=='classification'):
		pred = output.data.max(1)[1] # get the index of the max log-probability
	if(args.type=='regression'):
		pred = output.data.reshape((1,args.batchSz))[0].long()

	images = [transforms.ToPILImage()(image) for image in images]	
	images = [transforms.functional.affine(image,-angle,(0,0),1,0) for image,angle in zip(images,pred)]
	images = [transforms.ToTensor()(image) for image in images]	

	grid = utils.make_grid(images)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	plt.axis('off')
	plt.title(args.type+'\n'+str(labels.numpy())+'\n'+str(pred.numpy()))
	plt.show()  
	