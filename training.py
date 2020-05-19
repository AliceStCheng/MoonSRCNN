"""SRCNN network for moons; heavy influences/borrowing code from https://github.com/basher666/pytorch_srcnn
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch.nn
import torch.utils.data
import torch.optim
import torch.autograd
import torch.backends.cudnn as cudnn

from utils import AverageMeter, calc_psnr

import model
import pandas as pd
import data

import os
import shutil

import time

start_time = time.time()

torch.cuda.init()

print(torch.cuda.is_available())
print(torch.cuda.is_initialized())

cudnn.benchmark = True
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.current_device())
print('Using device:', device)
print()

# Check if GPU is used.
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


x = torch.rand(1000, 1000, 250)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


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
		total_loss += loss(model(input.to(device)), target.to(device))

	return total_loss/len(testing_data)



def train(epoch, training_data, model, lossfun, opt, total_epochs):
	"""Train the model and return the mean MSE over the training set.

	Args:
	:param epoch: Epoch number.
	:param training_data: DataLoader for the training dataset.
	:param model: Trained model to evaluate.
	:param lossfun: Loss function to use.
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
		predictions = model(input.to('cuda'))
		loss = lossfun(predictions, target.to('cuda'))
		total_loss += loss.data
		loss.backward()
		opt.step()

		# Update the user.
		progressbar('Epoch {:3d}/{:d},  Batch {:d}:  '.format(epoch,total_epochs,step),
			step, len(training_data),
			'  Losses: Total={:f}  Batch={:f}'.format(total_loss,loss.data))

	return total_loss/len(training_data)



def progressbar(prefix, x, y, postfix, symbol='■', length=32):
	a = int(length*x/y)
	b = length-a
	print('\r{} [{}{}] {}'.format(prefix,symbol*a,' '*b,postfix), end='\r')


def main():
	# Parse arguments.
	parser = argparse.ArgumentParser(description='MoonSRCNN Training')
	parser.add_argument('-u', dest='upscale', type=int, default=4, help='Super resolution upscale factor; default=4')
	parser.add_argument('-r', dest='seed', type=int, default=9823412, help='Random seed; default=9823412')
	parser.add_argument('-trb', dest='training_batch_size', type=int, default=32, help='Training batch size; default=64')
	parser.add_argument('-teb', dest='testing_batch_size', type=int, default=32, help='Testing batch size; default=64')
	parser.add_argument('-e', dest='num_epochs', type=int, default=30, help='Number of epochs to train for; default=10')
	parser.add_argument('-lr', dest='lr', type=float, default=0.0001, help='Learning Rate; default=0.01')
	parser.add_argument('-threads', dest='threads', type=int, default=2, help='Number of threads for the data loader; default=2')
	parser.add_argument('-name', dest='name', type=str , help='Name of the model', required=True)

	args = parser.parse_args()

	# directory for saving .pth files
	name = args.name
	output_dir = '/home/alice/alice/masters_project/MoonSRCNN-master/source/'+name+'/'

	# if os.path.exists(output_dir):
	# 	shutil.rmtree(output_dir)
	os.mkdir(output_dir)

	# Data loaders
	training_data = torch.utils.data.DataLoader(data.DatasetFromFolder('../dataset6/training/'),
				num_workers=args.threads, batch_size=args.training_batch_size,
				shuffle=True)
	testing_data = torch.utils.data.DataLoader(data.DatasetFromFolder('../dataset2/testing/'),
				num_workers=args.threads, batch_size=args.testing_batch_size,
				shuffle=False)

	# Setup the network, the loss function and the optimiser.
	moonsrcnn = model.MoonSRCNN()
	print("before: ", next(moonsrcnn.parameters()).is_cuda)
	moonsrcnn.to('cuda')
	print("after: ", next(moonsrcnn.parameters()).is_cuda)
	lossfun = torch.nn.MSELoss()
	opt = torch.optim.Adam(moonsrcnn.parameters(), lr=args.lr)

	# Train the network.
	training_loss_by_epoch = np.zeros(args.num_epochs)
	testing_loss_by_epoch = np.zeros(args.num_epochs)
	training_psnr_by_epoch = np.zeros(args.num_epochs) # this was added from MoonSRCNN
	testing_psnr_by_epoch = np.zeros(args.num_epochs) # this was added from MoonSRCNN

	for epoch in range(args.num_epochs):
		training_loss_by_epoch[epoch] = train(epoch, training_data, moonsrcnn, lossfun, opt, args.num_epochs)
		testing_loss_by_epoch[epoch] = test(testing_data, moonsrcnn, lossfun)
		# print(training_loss_by_epoch) <= prints the full array of training loss after each epoch
		# print(training_loss_by_epoch[epoch]) # <= prints training loss on its own after each epoch
		progressbar('Epoch {:3d}/{:d}:'.format(epoch+1,args.num_epochs),
			len(training_data), len(training_data),
			'  Losses: Training total={:f}  Testing total={:f}'.format(training_loss_by_epoch[epoch],testing_loss_by_epoch[epoch]))
		print('')
		print("--- %s seconds ---" % (time.time() - start_time))
		if epoch%2==0:
			torch.save(moonsrcnn, output_dir+'model%d.pth' %(epoch))

		epoch_psnr = AverageMeter()
		for datafgh in testing_data:
			inputs, labels = datafgh

			inputs = inputs.to(device)
			labels = labels.to(device)

			with torch.no_grad():
				preds = moonsrcnn(inputs).clamp(0.0, 1.0)

			epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

		testing_psnr_by_epoch[epoch] = epoch_psnr.avg

		epoch_psnr = AverageMeter()
		for datafgh in training_data:
			inputs, labels = datafgh

			inputs = inputs.to(device)
			labels = labels.to(device)

			with torch.no_grad():
				preds = moonsrcnn(inputs).clamp(0.0, 1.0)

			epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

		training_psnr_by_epoch[epoch] = epoch_psnr.avg

		print('eval psnr: {:.2f}'.format(epoch_psnr.avg))	

	print("--- %s seconds ---" % (time.time() - start_time))

	

	# can put this before loop and replot in each loop
	# figure = plt.figure()
	# plt.plot(np.arange(0,args.num_epochs), training_loss_by_epoch, label='Training')
	# plt.plot(np.arange(0,args.num_epochs), testing_loss_by_epoch, label='Testing')
	# plt.xlabel('Number of epochs')
	# plt.ylabel('Training loss')
	# figure.tight_layout()
	# # plt.show(block=False)
	# plt.savefig('../outputs/training_loss.png')

	df = pd.DataFrame(np.c_[training_loss_by_epoch, testing_loss_by_epoch, training_psnr_by_epoch, testing_psnr_by_epoch],
				columns=["trainingLoss", "testingLoss", "trainingPSNR", "testingPSNR"])

	df.to_csv("../outputs/df_"+name+".csv", index=False)

	f, (loss, psnr) = plt.subplots(1, 2, sharex=True)
	loss.set_ylabel("Loss")
	loss.set_xlabel("Number of epochs")
	# psnr.set_title("PSNR per epoch")
	psnr.set_ylabel("PSNR/dB")
	psnr.set_xlabel("Number of epochs")
	loss.plot(np.arange(0,args.num_epochs), training_loss_by_epoch, label='Training') # took out the minus 40 from the training loss
	loss.plot(np.arange(0,args.num_epochs), testing_loss_by_epoch, label='Testing') # took out the minus 40 from the training loss
	loss.legend()
	psnr.plot(np.arange(0,args.num_epochs), training_psnr_by_epoch, label='Training')
	psnr.plot(np.arange(0,args.num_epochs), testing_psnr_by_epoch, label='Testing')
	psnr.legend()
	f.tight_layout() # this and the following line hopefully solves the overlap issue.
	plt.savefig('../outputs/metrics_'+name+".png", dpi=100)
# plt.show(block=False)


	# data = np.array(training_loss_by_epoch)
	# data = data.T # transposes the data so you get 

    # name and path to where the file is saved.
	# path = './outputs/training_loss_and_psnr_%s.csv'
	# with open(path, 'w+') as datafile:
    #     np.savetxt(datafile, data)

	# np.savetext('training_loss.csv', training_loss_by_epoch, deliminiter='')

if __name__ == '__main__':
	main()
