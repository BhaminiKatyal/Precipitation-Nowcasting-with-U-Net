from glob import glob
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#pip install torchmetrics

from math import sqrt
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.regression import R2Score

# Path to the directory containing half-hourly NetCDF files
data_dir = '/content/drive/MyDrive/IMERG North India data (1st Jul - 15 Sep 2023)'
file_pattern = os.path.join(data_dir, '*.nc4*')

# Load all half-hourly NetCDF files into a single xarray Dataset
files = sorted(glob(file_pattern))
datasets = [xr.open_dataset(file, decode_times=False) for file in files]

# Concatenate the half-hourly datasets along the time dimension
combined_ds = xr.concat(datasets, dim='time')

# convert to numpy
np_dataset = np.array(combined_ds['precipitation'])

# plot some images
for i in range(0,len(np_dataset)-1000,4):
  plt.figure(figsize=(12,12))
  for j in range(4):
    plt.subplot(1,4,j+1)
    plt.imshow(np_dataset[i+j])

""" data preprocessing """

def crop_image(image,side):
  diff = image.shape[1] - side
  return image[:,diff//2:-diff//2,diff//2:-diff//2]

def crop_mask(mask,side):
  diff = mask.shape[1] - side
  return mask[diff//2:-diff//2,diff//2:-diff//2]

image_dataset = []
mask_dataset = []
for i in range(0,len(np_dataset)-3,4):
  image1 = torch.Tensor(np_dataset[i]).unsqueeze(0)
  image2 = torch.Tensor(np_dataset[i + 1]).unsqueeze(0)
  image3 = torch.Tensor(np_dataset[i + 2]).unsqueeze(0)

  # concatenate first 3 images
  image = torch.cat((image1, image2, image3), dim=0)

  # crop mask and image to 128x128
  image = crop_image(image,128)
  mask = crop_mask(np_dataset[i + 3],128)

  image_dataset.append(image)
  mask_dataset.append(torch.Tensor(mask))

print(len(image_dataset))
print(len(mask_dataset))

print(image_dataset[0].shape)
print(mask_dataset[0].shape)

class PrecipitationDataSet(Dataset):
  def __init__(self,image_dataset, mask_dataset):
    self.images = image_dataset
    self.masks = mask_dataset

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):

    image = self.images[index]
    mask = self.masks[index]

    # normalize image
    image = (image - image.min()) / (image.max() - image.min())
    return image,mask

dataset = PrecipitationDataSet(image_dataset, mask_dataset)

""" U-Net Model """

def double_conv(in_c,out_c):
  conv = nn.Sequential(
      nn.Conv2d(in_c,out_c,kernel_size = 3, stride = 1, padding='same', bias = False),
      nn.BatchNorm2d(out_c),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_c,out_c,kernel_size = 3, stride = 1, padding='same'),
      nn.BatchNorm2d(out_c),
      nn.ReLU(inplace=True)
  )
  return conv

def upscale(x,y):
  diffY = y.size()[2] - x.size()[2]
  diffX = y.size()[3] - x.size()[3]

  # Pad the input to match the size of the skip connection
  x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

  # Concatenate the skip connection with the upscaled input
  x = torch.cat([x, y], dim=1)
  return x

class unet(nn.Module):
  def __init__(self):
    super().__init__()

    self.pool = nn.MaxPool2d(2,2)
    self.down1 = double_conv(3,64)
    self.down2 = double_conv(64,128)
    self.down3 = double_conv(128,256)
    self.down4 = double_conv(256,512)

    self.up1 = nn.ConvTranspose2d(512,256,2,2)
    self.up_conv1 = double_conv(512,256)
    self.up2 = nn.ConvTranspose2d(256,128,2,2)
    self.up_conv2 = double_conv(256,128)
    self.up3 = nn.ConvTranspose2d(128,64,2,2)
    self.up_conv3 = double_conv(128,64)
    self.out = nn.Conv2d(64,1,1)

    self.initialize_weight

  def initialize_weight(self):
    for m in self.modules():
      if isinstance(m, nn.conv2d):
        nn.init.kaiming_uniform_(m.weight)

        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self,x):
    x1 = self.down1(x)
    x2 = self.pool(x1)
    x3 = self.down2(x2)
    x4 = self.pool(x3)
    x5 = self.down3(x4)
    x6 = self.pool(x5)
    x7 = self.down4(x6)

    x = self.up1(x7)
    x = upscale(x5,x)
    x = self.up_conv1(x)

    x = self.up2(x)
    x = upscale(x3,x)
    x = self.up_conv2(x)

    x = self.up3(x)
    x = upscale(x1,x)
    x = self.up_conv3(x)

    x = self.out(x)
    return x

model = unet()

""" Training the Model """

# setting up loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# divide data in train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = torch.utils.data.random_split(
    dataset, lengths=(train_size, test_size))

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

def train_unet(model, optimizer, train_loader):
  model.train()
  epoch_loss = 0.0
  count=0

  for images, masks in train_loader:
    images, masks = images.to(DEVICE), masks.to(DEVICE)
    count+=1
    model.zero_grad()
    output = model(images.cuda())
    loss = criterion(output.squeeze(), masks.cuda())
    epoch_loss += loss.item()
    loss.backward()
    optimizer.step()

  return epoch_loss

def test_unet(model, test_loader):
  model.eval()
  total_loss = 0.0
  for images, masks in test_loader:
    images, masks = images.to(DEVICE), masks.to(DEVICE)
    ouput = model(images.cuda())
    loss = criterion(ouput.squeeze(), masks.cuda())
    total_loss += loss.item()
  return total_loss

train_losses = []
test_losses = []
bestScore = inf
best_epoch = -1

for epoch in range(1,15):
  model.cuda()
  train_loss = train_unet(model, optimizer, train_loader)/4
  Current_loss = test_unet(model, test_loader)

  train_losses.append(train_loss)
  test_losses.append(Current_loss)

  print(f'Train Loss: {train_loss:.4f}')
  print(f'Test loss: {Current_loss:.4f}')

  # Plot Loss Curve
  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Training Loss')
  plt.plot(test_losses, label='Testing Loss')
  plt.title('Training Loss and Testing Loss')
  plt.legend()
  plt.show(block=False)

  if (Current_loss < bestScore):
    bestScore = Current_loss
    best_epoch = epoch
    torch.save(model.state_dict(), '/content/drive/MyDrive/unet_model.pth')
    print('Model Saved')

print(f'Best epoch: {best_epoch}')
print(f'Best loss: {bestScore}')

model.load_state_dict(torch.load('/content/drive/MyDrive/unet_model.pth'))

model.cuda()
model.eval()

""" Results """

# test data image prediction
for images,mask in test_loader:
  images = images.to(DEVICE)
  masks = mask.to(DEVICE)

  for i in range(images.shape[0]):
    plt.figure(figsize=(6,6))
    image = images[i]
    mask = masks[i]
    pred = model(image.unsqueeze(0).cuda())
    pred = pred.squeeze()

    # fit mask and pred in range 0 to 1
    mask = (mask-mask.min())/(mask.max()-mask.min())
    pred = (pred-pred.min())/(pred.max()-pred.min())

    #display image, actual mask and predicted mask
    plt.subplot( 1,3,1)
    image = image.cpu()
    mask = mask.cpu()
    plt.imshow(np.moveaxis(image.numpy(), 0, -1))
    plt.subplot( 1,3,2)
    plt.imshow(mask)
    plt.subplot( 1,3,3)
    pred = pred.squeeze()
    plt.imshow(pred.cpu().detach().numpy())
    plt.show()

""" Evaluation """

# fetch all pixels' prediction and label in seperate 1d arrays for evaluation

target = []
preds =[]

for images, masks in test_loader:
  images, masks = images.to(DEVICE), masks.to(DEVICE)

  for image, mask in zip(images, masks):
    # Loop through images in a batch

    output = model(image.unsqueeze(0).cuda())
    output = output.squeeze()

    mask = mask.flatten()
    output = output.flatten()
    mask = mask.to('cpu')
    output = output.to('cpu')

    for i in mask:
      target.append(i)

    for i in output:
      preds.append(i)

target = torch.tensor(target)
preds = torch.tensor(preds)

r2score = R2Score()
print("r2 score:", r2score(preds, target))

corr = PearsonCorrCoef()
print("Pearson Correlation Coefficient:", corr(preds, target))

def rmse_loss(preds, target):
  sumSquares = 0
  for i in range(len(preds)):
    sumSquares += (preds[i] - target[i])**2
  return sqrt(sumSquares/len(preds))

print("RMSE:", rmse_loss(preds, target))

