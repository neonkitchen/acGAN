import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import datetime
import os
from fid import *



def algo1_train(G, D_set, optimG, optimD, dataloader, dataloader_fid, criterion, z, N, T_max, T_w=None, alpha=0.01, _lambda=15, device = "cuda", save = True, vers = 0, batchSize = 128, fid=False, batchSizeFID = 128):
  '''
  - Implementation of primary Algorithm 1 for training Doan et al.'s Adaptative Curriculum GAN (acGAN) framework (https://arxiv.org/abs/1808.00020). 
  - Adapted from: https://github.com/Sirius79/acGAN/blob/master/train.py
  - Default parameters follow calibration in original paper. 
  
  * Inputs:
    - G: Generator
    - D_set: List of discriminators of varied capacity; len(D_list) == N
    - optimG: Optimiser for Generator
    - optimD: Optimiser list for D_set; len(optimD) == N
    - dataloader: dataloader
    - criterion: loss function
    - z: latent vector size
    - N: Number of discriminators
    - T_max: time steps
    - T_w: warm up time
    - alpha: reward moving average smoothing paramater
    - _lambda: Boltzmann constant
  
  * Returns:
    - img_list

  '''
  if T_w is None:
    T_w = 5 * N
    #T_w = 0
  
  ### Initialising...
  # Curriculum prob vector over all N D's
  pi = torch.full((N, ), 1/N, device=device)

  # Q values
  Q = torch.zeros(N, device=device)

  ### Setting empty lists for logging results
  if N==1:
    losses_D, losses_G = [[]], []
  elif N ==2:
    losses_D, losses_G = [[],[]], []
  elif N==3:
    losses_D, losses_G = [[],[],[]], []
  img_list = []
  fids = []
  pis = [] # list of pi arrays
  pis.append(pi)

  

  # Create batch of latent vectors tha will be used to visualize G progression
  fixed_noise = torch.randn(batchSize, z, 1, 1, device=device)
  print("Starting Training Loop...")
  begin_time = datetime.datetime.now()
  print(begin_time)

  ### Training loop 

  for t in range(T_max):
    

    # G_prev eval
    eval_noise = torch.randn(128, z, 1, 1, device=device)
    eval_fake = G(eval_noise)
    
    for i, data in enumerate(dataloader, 0):
      
      
      # real & fake labels
      real_label = random.uniform(0.7, 1.2)
      fake_label = random.uniform(0.0, 0.3)
      
  
      # dataloader itterates ([128, 3, 64, 64], 128 labels) tuple on defaults
      real_data = data[0].to(device) # [128, 3, 64, 64]
      batch_size = real_data.size(0) # 128
      label = torch.full((batch_size,), real_label, device=device) # 128 list of real_label

      ############################
      # (1) Update each discriminators D_{i} i=1...N as per Eq. 4: take gradient step to maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      for j, D in enumerate(D_set):
        
        ## Train with all-real batch
        D.zero_grad()
        output = D(real_data)

        errD_real = pi[j] * criterion(output, label)
        errD_real.backward()
        
         ## Train with all-fake batch
        noise = torch.randn(batch_size, z, 1, 1, device=device)
        fake = G(noise)
        label.fill_(fake_label)
        output = D(fake.detach())

        errD_fake = pi[j] * criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake

        losses_D[j].append(errD)
        optimD[j].step()
      
      ############################
      # (2) Update G network: take gradient step to maximize eqt 5: E[E[log(D(G(z)))]]
      ###########################
      label.fill_(real_label)  # fake labels are real for generator cost
      total_error = 0
      
      for j, D in enumerate(D_set):
        G.zero_grad()
        output = D(fake)
        errG = pi[j] * criterion(output, label)
        total_error += errG
      errG.backward()

      losses_G.append(errG)
      optimG.step()

       #Compute FID every epoch with batchSizeFID held-out samples at training time
      

      print('Done: [%d/%d][%d/%d]' % (t+1, T_max, i, len(dataloader)))

    ### End of epoch logging
    
              
    # Tracking policy to see if phase switching is observed, with each discriminator weight dominating a phase before eventually converging to uniform distribution.
    print('Saving losses & img_list @ epoch end...')
    torch.save(img_list, 'saved/img_list_v'+str(vers)+'.pt')
    torch.save(losses_D, 'saved/losses_D_v'+str(vers)+'.pt')
    torch.save(losses_G, 'saved/losses_G_v'+str(vers)+'.pt')
     # Appending pi just acted with
    pis.append(pi)
    print('Saving pis')
    torch.save(pis, 'saved/pis_v'+str(vers)+'.pt')

    print("Saving G's output on fixed_noise...")
    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    if fid:

      torch.save(fids, 'saved/fids_v'+str(vers)+'.pt')      print('Calculating FID...')
      with torch.no_grad():
          real_fid = next(iter(dataloader_fid))[0].detach().cpu()
          fake_fid = G(torch.randn(batchSizeFID, z, 1, 1, device=device)).detach().cpu()
          fid = calc_FID(real_fid, fake_fid)
          fids.append(fid)
      print('Saving FID...')
    
    if t >= T_w:
   
      print('Beyond T_w...')
      real_label = random.uniform(0.7, 1.2)
      label = torch.full((batchSize,), real_label, device=device)\


      # choose arm k 
      k = np.random.choice(N, 1, p=pi.cpu().numpy())[0]
      
      ### Evaluate performance of G with D_k...
      fake = G(eval_noise)
      output = D_set[k](fake.detach())
      
      ###... and observe reward for each discriminator i
      output_prev_G = D_set[k](eval_fake.detach())
      
      ### reward function that reflects progress made by G
      reward = criterion(output, label) - criterion(output_prev_G, label)
    
      ### Update all values in Q according to Eqt 2 & update pi values 
      Q[k] = alpha * reward + (1 - alpha) * Q[k]
      pi = F.softmax(_lambda * Q).detach().to(device)
      


      
      ### Averaging sample prs (pol) over every 20 databatch iterations
      #if i%int(200/batchSize) == 0:

      #if i%20 == 0:
      #print("Averaging sample probs (pol)")
      #for pi in pis: avg_pi += pi
      #avg_pi = (1/LEN)*avg_pi
      #avg_pis.append(avg_pi)


      
      # Resetting logs
      #pis = []
      #avg_pi = torch.full((N, ), 0, device=device)
      
      
     
  
  if save:

    torch.save(G, 'saved/G_v'+str(vers)+'.pt')
    torch.save(eval_fake, 'saved/eval_fake_v'+str(vers)+'.pt')
  
  print("Trainin done...") 
  print(datetime.datetime.now())
  print("Time elapsed during training...")  
  print(datetime.datetime.now() - begin_time)

  return losses_D, losses_G #, fids
