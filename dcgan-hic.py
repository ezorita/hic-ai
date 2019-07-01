import pdb
import sys
import pickle
import cooler
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

def is_power2(num):
	return num != 0 and ((num & (num - 1)) == 0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CollapseSC(nn.Module):
   def __init__(self, input_channels):
      super(CollapseSC, self).__init__()
      self.input_channels = input_channels
      self.collapse_layer = nn.ConvTranspose2d(input_channels, 1, kernel_size=3, stride=1, padding=1)
   def forward(self, x):
      h = self.collapse_layer(x).view((x.shape[0],x.shape[2],x.shape[3]))
      h = ((h.transpose(1,2) + h)/2).view(x.shape[0], 1, x.shape[3], x.shape[3])
      #return torch.bmm(h.transpose(1,2), h).view(x.shape[0], 1, x.shape[3], x.shape[3])
      return h

class ExpandSC(nn.Module):
   def __init__(self, output_channels):
      super(ExpandSC, self).__init__()
      self.output_channels = output_channels
      self.layer = nn.Conv2d(1, output_channels, kernel_size=1, stride=1, padding=0)

   def forward(self, x):
      return self.layer(x)


class DownSampler(nn.Module):
   def __init__(self, filt_size=5, stride=2, extra_padding=0):
      super(DownSampler, self).__init__()
      # Anti-aliasing filter coefficients (R.Zhang - Making convolutional network shift-invariant again)
      if(filt_size==1):
         a = np.array([1.,])
      elif(filt_size==2):
         a = np.array([1., 1.])
      elif(filt_size==3):
         a = np.array([1., 2., 1.])
      elif(filt_size==4):    
         a = np.array([1., 3., 3., 1.])
      elif(filt_size==5):    
         a = np.array([1., 4., 6., 4., 1.])
      elif(filt_size==6):    
         a = np.array([1., 5., 10., 10., 5., 1.])
      elif(filt_size==7):    
         a = np.array([1., 6., 15., 20., 15., 6., 1.])

      f = a[:,None]*a[None,:]
      self.register_buffer('filt_weights', torch.Tensor(f/f.sum())[None,None,:,:])
      
      pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
      self.pad = nn.ReflectionPad2d(pad_sizes)
      self.stride = stride

   def forward(self, x):
      xpad = self.pad(x)
      xpad_singlechan = xpad.view(-1,1,xpad.shape[2],xpad.shape[3])
      xdown_sc = func.conv2d(xpad_singlechan, self.filt_weights, stride=self.stride) 
      return xdown_sc.view(x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2)

   
class DoubleResolution(nn.Module):
   def __init__(self, in_channels, norm=True):
      super(DoubleResolution, self).__init__()
      
      self.layers = nn.Sequential(
         nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=4, stride=2, padding=1),
         nn.BatchNorm2d(in_channels//2) if norm else lambda x: x
      )

   def forward(self, x):
      return self.layers(x)

   
class HalfResolution(nn.Module):
   def __init__(self, in_channels, norm=True):
      super(HalfResolution, self).__init__()
      
      self.layers = nn.Sequential(
         nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=4, stride=2, padding=1),
         nn.BatchNorm2d(in_channels*2) if norm else lambda x: x
      )

   def forward(self, x):
      return self.layers(x)

   
class HiCNNDiscriminator(nn.Module):
   def __init__(self, nfeatures, device, resolution=1024, dropout=.3):
      super(HiCNNDiscriminator, self).__init__()
      
      # Check output resolution
      assert is_power2(resolution)

      # Number of downsamplers
      n_blocks = 0
      while 2**(n_blocks+2) < resolution:
         n_blocks += 1

      self.device = device
      self.nfeatures = nfeatures
      self.n_blocks = 0
      self.max_blocks = n_blocks

      ## Layers

      # Channel expanders (From image to nfeatures)
      self.expand = nn.ModuleList()
      nfeat = self.nfeatures
      for i in np.arange(self.max_blocks+1):
         self.expand.append(ExpandSC(nfeat))
         nfeat = nfeat//2
      
      # Downsamplers (2D layers)
      self.downsamplers = nn.ModuleList()
      nfeat = self.nfeatures
      for i in np.arange(n_blocks):
         self.downsamplers.append(HalfResolution(nfeat//2))
         nfeat = nfeat//2

      # Final layer, 2D to 1D discriminator decision
      self.img_to_dis = nn.Sequential(
         nn.Conv2d(in_channels=self.nfeatures, out_channels=1, kernel_size=4, stride=1, padding=0),
         nn.Sigmoid()
      )

      # Move to device
      self.expand = self.expand.to(self.device)
      self.downsamplers = self.downsamplers.to(self.device)
      self.img_to_dis = self.img_to_dis.to(self.device)
      
   def forward(self, x):
      x = self.expand[self.n_blocks](x)
      i = self.n_blocks
      while i > 0:
         i -= 1
         x = self.downsamplers[i](x)
      x = self.img_to_dis(x)
      return x


class HiCNNGenerator(nn.Module):
   def __init__(self, lat_sz,nfeatures, device, resolution=1024, dropout=0.3):
      super(HiCNNGenerator, self).__init__()
      
      # Check output resolution
      assert is_power2(resolution)

      # Number of upsamplers
      n_blocks = 0
      while 2**(n_blocks+2) < resolution:
         n_blocks += 1

      self.latent_size = lat_sz
      self.device = device
      self.nfeatures = nfeatures
      self.n_blocks = 0
      self.max_blocks = n_blocks

      ## Layers

      # Latent to image channels
      self.lat_to_img = nn.Sequential(
         nn.ConvTranspose2d(in_channels=lat_sz, out_channels=self.nfeatures, kernel_size=4, stride=1, padding=0),
         nn.BatchNorm2d(self.nfeatures),
         nn.ReLU()
      )

      # Upsamplers (2D layers)
      self.upsamplers = nn.ModuleList()
      nfeat = self.nfeatures
      for i in np.arange(n_blocks):
         self.upsamplers.append(DoubleResolution(nfeat))
         nfeat = nfeat//2

      # Channel collapse (generate single channel image form multiple channels)
      self.collapse = nn.ModuleList()
      nfeat = self.nfeatures
      for i in np.arange(self.max_blocks+1):
         self.collapse.append(CollapseSC(nfeat))
         nfeat = nfeat//2

      # Move to device
      self.lat_to_img = self.lat_to_img.to(self.device)
      self.upsamplers = self.upsamplers.to(self.device)
      self.collapse = self.collapse.to(self.device)
      
   def forward(self, z):
      z = self.lat_to_img(z)
      for i in np.arange(self.n_blocks):
         z = self.upsamplers[i](z)
      # Collapse channels
      z = self.collapse[self.n_blocks](z)
      return torch.sigmoid(z)


class HiCNN():
   def __init__(self, lat_sz, nfeatures, device, output_resolution=1024, dropout=0.3):
      # store args
      self.device = device
      self.latent_size = lat_sz
      
      # Networks
      self.generator = HiCNNGenerator(lat_sz, nfeatures, device)
      self.discriminator = HiCNNDiscriminator(nfeatures, device)

     

   def increase_resolution(self):
      self.generator.n_blocks = min(self.generator.n_blocks+1, self.generator.max_blocks)
      self.discriminator.n_blocks = min(self.discriminator.n_blocks+1, self.discriminator.max_blocks)


   def optimize(self, train_data, test_data, batch_size, batches=1000, batches_per_step=100, z_dist=torch.randn, dis_lr=0.001, gen_lr=0.0002):

      # 256x256 resolution
      self.generator.n_blocks = self.generator.max_blocks - 2
      self.discriminator.n_blocks = self.discriminator.max_blocks - 2
      
      # Parameters
      sample_period = 10

      # Optimizers
      gen_opt = torch.optim.Adam(self.generator.parameters(), lr=gen_lr, betas=(0.5, 0.9))
      dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.9))

      # Discriminator targets
      dis_accept = torch.ones(batch_size).to(self.device)
      dis_reject = torch.zeros(batch_size).to(self.device)

      # Loss log
      gen_log = []
      dis_log = []

      # Training progress figure
      figstat = None

      # Discriminator Learn Factor
      mean_buf = 100
      true_acc = np.nan * np.zeros(mean_buf)
      fake_acc = np.nan * np.zeros(mean_buf)
      fool_acc = np.nan * np.zeros(mean_buf)
      dis_loss = np.nan * np.zeros(mean_buf)
      gen_loss = np.nan * np.zeros(mean_buf)

      # Downsampler
      downsampler = DownSampler().to(self.device)

      for e in np.arange(batches):
         ###
         ## Generate random batch
         ###
         dis_real = random_batch(train_data, batch_size).to(self.device)

         # Downscale image to current resolution
         for i in np.arange(self.generator.max_blocks - self.generator.n_blocks):
            dis_real = downsampler(dis_real)

         # Normalize max value to 1.0
         dis_real_ = dis_real.view(batch_size, -1)
         dis_real = (dis_real_.transpose(0,1)/dis_real_.max(1).values).transpose(0,1).view(dis_real.shape)

         # Generator sample
         z_sample = z_dist((batch_size, self.latent_size, 1, 1)).to(self.device)

         ###
         ## Train discriminator
         ###
         with torch.no_grad():
            # Fake samples
            dis_fake = self.generator(z_sample)
         # Loss on Real samples
         dis_out_real = self.discriminator(dis_real).squeeze()
         loss_real = nn.functional.binary_cross_entropy(dis_out_real, dis_accept, reduction='mean')
         # Loss on Fake samples
         dis_out_fake = self.discriminator(dis_fake).squeeze()
         loss_fake = nn.functional.binary_cross_entropy(dis_out_fake, dis_reject, reduction='mean')
         # Disciminator Loss
         dloss = .5*(loss_real + loss_fake)
         # Update parameters
         gen_opt.zero_grad()
         dis_opt.zero_grad()
         dloss.backward()
         dis_opt.step()
         # Discriminator accuracy (true/fake)
         t_acc = float(torch.sum(dis_out_real.detach().cpu().view(-1) > torch.rand(batch_size)))/batch_size
         f_acc = float(torch.sum(dis_out_fake.detach().cpu().view(-1) < torch.rand(batch_size)))/batch_size
         # Log values
         dis_loss[e%mean_buf] = float(dloss)
         true_acc[e%mean_buf] = t_acc
         fake_acc[e%mean_buf] = f_acc

         ###
         ## Train generator
         ###
         z_sample = z_dist((batch_size, self.latent_size, 1, 1)).to(self.device)
         dis_fool = self.discriminator(self.generator(z_sample)).squeeze()
         gloss = nn.functional.binary_cross_entropy(dis_fool, dis_accept)
         # Update parameters
         gen_opt.zero_grad()
         dis_opt.zero_grad()
         gloss.backward()
         gen_opt.step()
         # Log loss
         # Accuracy
         facc = float(torch.sum(dis_fool.detach().cpu() > torch.rand(batch_size)))/batch_size
         # Log values
         gen_loss[e%mean_buf] = float(gloss)
         fool_acc[e%mean_buf] = facc

         # Increase resolution
         #if (e+1) % batches_per_step == 0:
         #   self.increase_resolution()

         # Sample images
         if (e+1) % sample_period == 0:
            plt.imshow(np.log1p(dis_fake[np.random.randint(batch_size),0,:,:].cpu()), cmap=plt.cm.Reds)
            plt.savefig('sample_images/sample_hic_batch_{}_res_{}.png'.format(e+1,dis_real.shape[-1]))
            plt.close()

         # Verbose
         print("batch: {}, resolution: {}x{} (upsamplers: {}), genLoss: {:.3E} ({:.2f}%), disLoss: {:.3E} (true: {:.2f}%, fake: {:.2f}%)".format(e+1,dis_real.shape[-1], dis_real.shape[-1], self.generator.n_blocks, np.nanmean(gen_loss), np.nanmean(fool_acc)*100, np.nanmean(dis_loss), np.nanmean(true_acc)*100, np.nanmean(fake_acc)*100))

         
def img_from_diags(positions, diags):
   size = diags.shape[1]
   img = torch.zeros((len(positions),1,size,size))
   for n, pos in enumerate(positions):
      for i,p in enumerate(np.arange(pos, pos+size)):
         img[n,0,i,i:] = diags[p,:size-i]
         img[n,0,i+1:,i] = diags[p,1:size-i]

   return img

def random_batch(data, batch_size):
   # Sampling probabilities
   c_idx = [(c[:-c.shape[1],0] > 0).nonzero().view(-1).numpy() for c in data]
   samp_p = np.array([len(idx) for idx in c_idx])
   samp_p = samp_p/np.sum(samp_p)

   # Samples per chromosome
   chr_samps = np.array(
      np.unique(
         np.random.choice(len(c_idx), size=batch_size, replace=True, p=samp_p),
         return_counts=True
      )
   ).T

   # Generate random samples
   resolution = data[0].shape[1]
   batch = torch.zeros((0,1,resolution,resolution))
   for c, n_samp in chr_samps:
      chr_data = data[c]
      chr_pos = np.random.choice(c_idx[c], replace=False, size=n_samp)
      chr_batch = img_from_diags(chr_pos, chr_data)
      batch = torch.cat((batch, chr_batch), 0)

   return batch

if __name__ == "__main__":
   # with open('ai_dataset_5kb.pickle', 'rb') as f:
   #    chr_seqs = pickle.load(f)

   cf = cooler.Cooler('hic_datasets/WT_hg19_5k_q10.cool')
   hmat = cf.matrix(balance=False)
   hic_range = 1024

   #train_chr = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10']
   train_chr = ['chr9','chr10']
   train_hic = []
   for c in train_chr:
      # Extract HiC diagonals
      hic_mat = hmat.fetch(c)
      hic_len = hic_mat.shape[0]
      hic_diags = np.ones((hic_len, hic_range)) * np.nan
      for i in np.arange(hic_range):
         hic_diags[:(hic_len-i),i] = np.diagonal(hic_mat, i)

      # Normalize
      NORM_QUANTILE = 0.95
      # Diagonal decay normlization
      #norm_fact = np.nanmedian(hic_diags, 0)
      #norm_fact[np.where(norm_fact < 1)] = 1
      hic_diags_norm = hic_diags#/norm_fact
      # Max value normalization
      hic_diags_norm = hic_diags_norm/np.quantile(hic_diags_norm[~np.isnan(hic_diags_norm)], NORM_QUANTILE)
      hic_diags_norm[np.where(hic_diags_norm > 1.0)] = 1.0
      
      # Append data
      train_hic.append(torch.Tensor(hic_diags_norm))

   lat_sz = 100
   k = 50
   nfeat  = 512
   batches = 20000
   batches_per_step = 100
   batch_size = 32
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   hicnn = HiCNN(lat_sz, nfeat, device, output_resolution=hic_range)

   dis_lr = 0.0001
   gen_lr = 0.0001
   hicnn.optimize(train_hic, None, batch_size, batches=batches, batches_per_step=batches_per_step, dis_lr=dis_lr, gen_lr=gen_lr)

