'''
 - Adapted from: https://github.com/Sirius79/acGAN/blob/master/main.py
 - Default parameters follow calibration in original paper. 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from models import *
from train_v1 import algo1_train
### Packages needed for WandB logging !wandb login
#import logging
#logging.propagate = False 
#logging.getLogger().setLevel(logging.ERROR)
#import wandb



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--batchSizeFID', type=int, default=128, help='FID score batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--z', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of the generator features')
parser.add_argument('--ndf', type=int, default=64, help='size of the discriminator features')
parser.add_argument('--N', type=int, default=1, help='number of discriminators')
#parser.add_argument('--N', type=int, default=2, help='number of discriminators') # N = 2 does not produce high fedility images
#parser.add_argument('--N', type=int, default=3, help='number of discriminators') # N = 3 does not produce high fedility images
parser.add_argument('--T_max', type=int, default=25, help='number of time steps to train for (epochs)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimiser, default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimiser, default=0.999')
parser.add_argument('--save', type=bool, default=True, help='Save to file, default=True')
parser.add_argument('--fid', type=bool, default=False, help='FID score calculated @ end of each epoch, default=False')
parser.add_argument('--vers', type=bool, default=1, help='vers, default=1')
parser.add_argument('--model', type=int, default=1, help='model, default=1:acGAN, 2:GMAN-1; 3: Uniform')
parser.add_argument('--seed', type=int, default=0, help='Set specific seed for training; defaul randomly selects seed')
# Check device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("Running on device...", device)
 
# Parsing in arguments 
opt = parser.parse_args()

### Setting random seed for run
if opt.seed == 0:
  manualSeed = random.randint(1, 10000) # use if you want new results
else:
  manualSeed = opt.seed

print("Manual Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if opt.save == True:
  torch.save(manualSeed, 'saved/manualSeed_v'+str(opt.vers)+'.pt')


# Calibrating based on model (used for benchmarking)
if opt.model == 1:
  # acGAN
  alpha=0.01
  _lambda=15
elif opt.model ==2:
  # GMAN-1
  alpha=1
  _lambda=1
elif opt.model ==3:
  # Uniform
  alpha=0.01
  _lambda=0


### Loading dataset... 

# load data
if opt.dataset == 'mnist':
  dataset = dataset.MNIST(root='../../data/', download=True, train=True,
                       transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5), (0.5))]))
  ch = 1
elif opt.dataset == 'cifar10':
  dataset = dataset.CIFAR10(root='../../data/', download=True, train=True,\
                         transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
  ch = 3


dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

# Calibratring seperate dataloader for FID metric
dataloader_fid = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSizeFID, shuffle=True)

if opt.save == True:
  print('Saving dataloders...')
  torch.save(dataloader, 'saved/dataloader_v'+str(opt.vers)+'.pt')
  #torch.save(dataloader_fid, 'saved/dataloader_fid_v'+str(opt.vers)+'.pt')


# Setting vector sizes
z = int(opt.z)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
T_max = int(opt.T_max)
N = int(opt.N)

### Intialising Generator
DCG = DCGenerator(c=ch).to(device)
# custom weights initialization called 
DCG.apply(weights_init)

### Setting optimiser for Generator
optimG = optim.Adam(DCG.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))
 
### Intialising 3 Discriminators
DCD1 = DCDiscriminator1(c=ch).to(device)
# custom weights initialization called 
DCD1.apply(weights_init)

DCD2 = DCDiscriminator2(c=ch).to(device)
# custom weights initialization called 
DCD2.apply(weights_init)

DCD3 = DCDiscriminator3(c=ch).to(device)
# custom weights initialization called 
DCD3.apply(weights_init)

### Setting optimiser for Discriminators
optimizerD1 = optim.Adam(DCD1.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))
optimizerD2 = optim.Adam(DCD2.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))
optimizerD3 = optim.Adam(DCD3.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))

### Setting D_set & optimizer_set
if N==1:
  D_set = [DCD1]
  optimD = [optimizerD1]
elif N==2:
  D_set = [DCD1, DCD2]
  optimD = [optimizerD1, optimizerD2]
elif N==3:
  D_set = [DCD1, DCD2, DCD3]
  optimD = [optimizerD1, optimizerD2, optimizerD3]



### Setting Loss Criterion
criterion = nn.BCELoss()

### Trainining using Algorithm 1
loss_d, loss_g = algo1_train(G = DCG,
                            D_set = D_set,
                            optimG = optimG,
                            optimD = optimD,
                            dataloader = dataloader,
                            dataloader_fid = dataloader_fid,
                            criterion = criterion,
                            z = z,
                            N = N,
                            T_max = T_max, 
                            T_w = None,
                            alpha = alpha,
                            _lambda = _lambda,
                            device = device, 
                            save = True,
                            vers = opt.vers,
                            batchSize = opt.batchSize,
                            fid = opt.fid,
                            batchSizeFID = opt.batchSizeFID)

if opt.save == True:
  print('Saving models...')
  torch.save(DCG, 'saved/DCG_v'+str(opt.vers)+'.pt')
  torch.save(DCD1, 'saved/DCD1_v'+str(opt.vers)+'.pt')
  torch.save(DCD2, 'saved/DCD2_v'+str(opt.vers)+'.pt')
  torch.save(DCD3, 'saved/DCD3_v'+str(opt.vers)+'.pt')
  print('TRAINING COMPLETE :)')
