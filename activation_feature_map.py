import torch

import matplotlib.pyplot as plt
import data
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('/home/alice/alice/masters_project/MoonSRCNN-master/source/30e_64arch_Adam_MSE_0001lr/model28.pth')

input = data.load_input('/home/alice/alice/masters_project/MoonSRCNN-master/dataset2/validation/0006_hr.png', 1).to(device)
input = torch.reshape(input, (1,1,input.shape[1],input.shape[2])).to(device)

# build feature map (feature map = activation map: output of first layer for a
# given picture)
print(input.shape)
a1 = model._conv1(input)
print(a1.shape)
fig, axs = plt.subplots(8, 8, figsize=(20, 20))
axs = axs.flatten()
for i in range(len(axs)):
  ax = axs[i]
  ax.imshow(a1.cpu().detach().numpy()[0,i,:,:], cmap='Greys')
  ax.set_xticks([])
  ax.set_yticks([])
fig.tight_layout()
#fig.savefig('/home/alice/alice/masters_project/MoonSRCNN-master/outputs/activation1.png')
# plt.show()

# showing filters
v_min = np.min(model._conv1.weight.cpu().detach().numpy())
v_max = np.max(model._conv1.weight.cpu().detach().numpy())
fig, axs = plt.subplots(8, 8, figsize=(20, 20))
axs = axs.flatten()
for i in range(len(axs)):
  ax = axs[i]
  ax.imshow(model._conv1.weight.cpu().detach().numpy()[i,0,:,:], cmap='Greys', vmin=v_min, vmax=v_max)
  ax.set_xticks([])
  ax.set_yticks([])
#fig.savefig('/home/alice/alice/masters_project/MoonSRCNN-master/outputs/weights1.png')
plt.show()


print(input.shape)
out = model._relu1(a1)
a2 = model._conv2(out)
#print(a1.shape)

# showing the filters
fig, axs = plt.subplots(8, 4, figsize=(10, 20))
axs = axs.flatten()
for i in range(len(axs)):
  ax = axs[i]
  ax.imshow(a2.cpu().detach().numpy()[0,i,:,:], cmap='Greys')
  ax.set_xticks([])
  ax.set_yticks([])
fig.tight_layout()
#fig.savefig('/home/alice/alice/masters_project/MoonSRCNN-master/outputs/activation2.png')
# plt.show()

# showing weights
fig, ax = plt.subplots()
print(model._conv2.weight.cpu().detach().numpy().shape)
ax.imshow(model._conv2.weight.cpu().detach().numpy()[:,:,0,0], cmap='Greys')
ax.set_xticks([])
ax.set_yticks([])
#fig.savefig('/home/alice/alice/masters_project/MoonSRCNN-master/outputs/weights2.png')
# plt.show()

# showing the bias
fig, ax = plt.subplots()
print(model._conv2.bias.cpu().detach().numpy().shape)
ax.imshow(model._conv2.bias.cpu().detach().numpy()[:,:,0,0], cmap='Greys')
ax.set_xticks([])
ax.set_yticks([])
#fig.savefig('/home/alice/alice/masters_project/MoonSRCNN-master/outputs/weights2.png')
plt.show()



