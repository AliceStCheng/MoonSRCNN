import torch

# Copy of SRCNN network from https://github.com/basher666/pytorch_srcnn
class MoonSRCNN(torch.nn.Module):
	def __init__(self):
		super(MoonSRCNN,self).__init__()
		self._conv1 = torch.nn.Conv2d(1, 64, kernel_size=9, padding=4)
		self._relu1 = torch.nn.ReLU()
		self._conv2 = torch.nn.Conv2d(64, 32, kernel_size=1, padding=0)
		self._relu2 = torch.nn.ReLU()
		self._conv3 = torch.nn.Conv2d(32, 1, kernel_size=5, padding=2)

	def forward(self, x):
		out = self._conv1(x)
		out = self._relu1(out)
		out = self._conv2(out)
		out = self._relu2(out)
		out = self._conv3(out)

		return out
