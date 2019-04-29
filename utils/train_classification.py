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
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		nProcessed += len(data)
		pred = output.data.max(1)[1] # get the index of the max log-probability
		incorrect = pred.ne(target.data).cpu().sum()
		err = 100.*incorrect/len(data)
		partialEpoch = epoch + batch_idx / len(trainLoader) - 1
		print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
			partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
			loss.data[0], err))
		trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
		trainF.flush()