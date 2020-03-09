import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

import matplotlib.pyplot as plt
import numpy as np

import time

start_time = time.time()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3) # originallly 3
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8) # originally 16, reduced to improve generalisation ability.
    parser.add_argument('--num-epochs', type=int, default=10) # originally 400
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion =  nn.SmoothL1Loss() # used to be nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    training_loss_by_epoch = np.zeros(args.num_epochs) # this was added from MoonSRCNN
    #testing_loss_by_epoch = np.zeros(args.num_epochs)
    training_psnr_by_epoch = np.zeros(args.num_epochs) # this was added from MoonSRCNN

    for epoch in range(args.num_epochs):
        model.train()
        #testing_loss_by_epoch[epoch] = test(testing_data, model, criterion)
        epoch_losses = AverageMeter()
        # print(epoch_losses)

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            # changed from .format(epoch, args.num_epochs - 1) so that it
            # matches the same number of epochs as entered in the terminal
            t.set_description('epoch: {}/{}'.format(epoch + 1, args.num_epochs))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        # print(model.state_dict()) <= prited to see what this does - I think it's responsible for the
        # intermediate trained images.
        model.eval()
        # print(model.eval()) <= this prints outs the architecture of the cnn.
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        # print('{}'.format(epoch_psnr.avg))   <= to just print out the psnr value to 16 s.f.
        # print('{}'.format(epoch_losses.avg))   <= to just print out the loss value to 16 s.f.

        training_loss_by_epoch[epoch] = epoch_losses.avg
        training_psnr_by_epoch[epoch] = epoch_psnr.avg

        #print(training_loss_by_epoch) #<= prints the list of training losses so far for this epoch
        #print(training_psnr_by_epoch) #<= prints the list of training psnr so far for this epoch

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

    print("--- %s seconds ---" % (time.time() - start_time))

    # save the training loss and the PSNR values in one file

    data = np.array([training_loss_by_epoch, training_psnr_by_epoch])
    data = data.T # transposes the data so you get two columns

    # name and path to where the file is saved.
    path = './outputs/training_loss_and_psnr.csv'
    with open(path, 'w+') as datafile:
        np.savetxt(datafile, data, fmt=['%.14f', '%.14f'], delimiter=",")

    # Plot the training loss and the PSNR

    f, (training_loss, psnr) = plt.subplots(1, 2, sharex=True)
    training_loss.set_title("training loss per epoch")
    training_loss.set_ylabel("training loss")
    training_loss.set_xlabel("number of epochs")
    psnr.set_title("PSNR per epoch")
    psnr.set_ylabel("training psnr")
    psnr.set_xlabel("number of epochs")
    training_loss.plot(np.arange(0,args.num_epochs), training_loss_by_epoch - 40, label='Training')
    psnr.plot(np.arange(0,args.num_epochs), training_psnr_by_epoch - 40, label='PSNR')
    plt.savefig('training_metrics.png', dpi=100)
    plt.show()
