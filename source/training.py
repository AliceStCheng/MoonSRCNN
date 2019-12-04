"""SRCNN network for moons; heavy influences/borrowing code from https://github.com/basher666/pytorch_srcnn
"""

import matplotlib.pyplot as pl
import numpy as np
import argparse

import torch.nn
import torch.utils.data
import torch.optim
import torch.autograd

import model
import data

def test(testing_data, model, loss):
	"""Test the model again the test data set and return the mean MSE.

	Args:
	:param testing_data: DataLoader for the test dataset.
	:param model: Trained model to evaluate.
	:param loss: Loss function to use.
	"""
	total_loss = 0
	for batch in testing_data:
		# Get a batch of samples from the testing dataset.
		input, target = torch.autograd.Variable(batch[0]), torch.autograd.Variable(batch[1])

		# Compute the loss function and add to the total loss.
		total_loss += loss(model(input), target)

	return total_loss/len(testing_data)



def train(epoch, training_data, model, loss, opt, total_epochs):
	"""Train the model and return the mean MSE over the training set.

	Args:
	:param epoch: Epoch number.
	:param training_data: DataLoader for the training dataset.
	:param model: Trained model to evaluate.
	:param loss: Loss function to use.
	:param opt: Optimiser to use.
	"""
	total_loss = 0

	for step, batch in enumerate(training_data, 1):
		# Get a batch of samples.
		input, target = torch.autograd.Variable(batch[0]), torch.autograd.Variable(batch[1])

		# Clear gradients
		opt.zero_grad()

		# Compute model predictions, the loss function, add the loss to our
		# running total, compute gradients and then take a step.
		predictions = model(input)
		loss = lossfun(predictions, target)
		total_loss += loss.data
		loss.backward()
		opt.step()

		# Update the user.
		progressbar('Epoch {:3d}/{:d}:'.format(epoch,total_epochs),
			step, len(training_data),
			'  Losses: Total={:f}  Batch={:f}'.format(total_loss,loss.data))

	return total_loss/len(training_data)



def progressbar(prefix, x, y, postfix, symbol='■', length=32):
	a = int(length*x/y)
	b = length-a
	print('\r{} [{}{}] {}'.format(prefix,symbol*a,' '*b,postfix), end='\r')



if __name__ == '__main__':
	# Parse arguments.
	parser = argparse.ArgumentParser(description='MoonSRCNN Training')
	parser.add_argument('-u', dest='upscale', type=int, default=4, help='Super resolution upscale factor; default=4')
	parser.add_argument('-r', dest='seed', type=int, default=9823412, help='Random seed; default=9823412')
	parser.add_argument('-trb', dest='training_batch_size', type=int, default=64, help='Training batch size; default=64')
	parser.add_argument('-teb', dest='testing_batch_size', type=int, default=64, help='Testing batch size; default=64')
	parser.add_argument('-e', dest='num_epochs', type=int, default=10, help='Number of epochs to train for; default=10')
	parser.add_argument('-lr', dest='lr', type=float, default=0.01, help='Learning Rate; default=0.01')
	parser.add_argument('-threads', dest='threads', type=int, default=2, help='Number of threads for the data loader; default=2')
	args = parser.parse_args()

	# Data loaders
	training_data = torch.utils.data.DataLoader(data.DatasetFromFolder('../dataset/training/'),
				num_workers=args.threads, batch_size=args.training_batch_size,
				shuffle=True)
	testing_data = torch.utils.data.DataLoader(data.DatasetFromFolder('../dataset/testing/'),
				num_workers=args.threads, batch_size=args.testing_batch_size,
				shuffle=False)

	# Setup the network, the loss function and the optimiser.
	moonsrcnn = model.MoonSRCNN()
	lossfun = torch.nn.MSELoss()
	opt = torch.optim.Adam(moonsrcnn.parameters(), lr=args.lr)

	# Train the network.
	training_loss_by_epoch = np.zeros(args.num_epochs)
	testing_loss_by_epoch = np.zeros(args.num_epochs)
	for epoch in range(args.num_epochs):
		training_loss_by_epoch[epoch] = train(epoch, training_data, moonsrcnn, lossfun, opt, args.num_epochs)
		testing_loss_by_epoch[epoch] = test(testing_data, moonsrcnn, lossfun)
		progressbar('Epoch {:3d}/{:d}:'.format(epoch+1,args.num_epochs),
			len(training_data), len(training_data),
			'  Losses: Training total={:f}  Testing total={:f}'.format(training_loss_by_epoch[epoch],testing_loss_by_epoch[epoch]))
		print('')
		if epoch%2==0:
			torch.save(moonsrcnn, 'model.pth')

	pl.plot(np.arange(0,args.num_epochs), training_loss_by_epoch, label='Training')
	pl.plot(np.arange(0,args.num_epochs), testing_loss_by_epoch, label='Testing')
	pl.show()
