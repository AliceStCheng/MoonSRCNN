import torch
from torch import nn

# Copy of SRCNN network from https://github.com/basher666/pytorch_srcnn
class MoonSRCNN(torch.nn.Module):
	def __init__(self):
		super(MoonSRCNN,self).__init__()
		self._conv1 = torch.nn.Conv2d(1, 64, kernel_size=9, padding=9//2) # used to be Conv2d(1, 64, kernel_size=9, padding=9//2)
		self._relu1 = torch.nn.ReLU()
		self._conv2 = torch.nn.Conv2d(64, 32, kernel_size=5, padding=5//2) # used to be Conv2d(64, 32, kernel_size=9, padding=9//2)
		self._relu2 = torch.nn.ReLU()
		self._conv3 = torch.nn.Conv2d(32, 1, kernel_size=5, padding=5//2) # used to be Conv2d(32, 1, kernel_size=9, padding=9//2)

	def forward(self, x):
		out = self._conv1(x)
		out = self._relu1(out)
		out = self._conv2(out)
		out = self._relu2(out)
		out = self._conv3(out)

		return out

	# def forward(self, x):
	#     x = self.relu(self.conv1(x))
	#     x = self.relu(self.conv2(x))
	#     x = self.conv3(x)
	#     return x
