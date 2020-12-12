import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import helper
import sys,os,signal
from time import time
import numpy as np
from torch import Tensor
import pandas
import re
from skimage import io
from torch.utils.data import Dataset, DataLoader

##########################
### import Data
##########################
path = ... 
stats = pandas.read_csv("sango.csv")
stats=stats.dropna()   # remove characters without stats
stats["Sex"]=(stats["Sex"]=="男")*1.0  # 1=male, 0=female. convert to double
imgnamelist=os.listdir("images")
# construct image-address look-up
imgstat = {}
for imgname in imgnamelist:
  # catch until last consecutive chinese char
  personname=imgname[:re.search(u'[^\u4E00-\u9FA5]',imgname).start()] 
  if personname in stats["Name"].values:
    # path as key because some characters have multiple pictures
    imgstat[imgname]=personname  
# build address book 
imgstat=pandas.DataFrame(list(imgstat.items()), columns=["imgpath","personname"]) 
# only keep matched names
imgstat=pandas.merge(imgstat,stats,how="inner",left_on="personname",right_on="Name")


class SGZ(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(64)])
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        image = self.transform(io.imread(os.path.join("SGZ/images",self.df.imgpath[index])))
        label = self.df[["Sex","LEAD","MAR","INT","POL","CHAR"]].iloc[index].to_numpy()/100
        return image, label



##########################
### Define Models 
##########################
class Block(torch.nn.Module):
    def __init__(self, filters):
        super(Block, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm3d(filters), torch.nn.ReLU(),
            torch.nn.Conv3d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm3d(filters))

    def forward(self, x):
        return F.relu(x + self.block(x))

class DBlock(torch.nn.Module):
    def __init__(self, filters, stride=1):
        super(DBlock, self).__init__()
        self.stride=stride

        # No BatchNorm
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(filters, filters, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv3d(filters, filters, 3, padding=1, stride=stride, bias=False))

    def forward(self, x):
        return F.relu(x[:,:,:,::self.stride,::self.stride] + self.block(x))

class Upsample(torch.nn.Module):
    def __init__(self, fin, fout, factor):
        super(Upsample, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Upsample(size=[3, 64, 64], mode='trilinear', align_corners=False),
            torch.nn.Conv3d(fin, fout, 3, padding=1, bias=False),
            torch.nn.BatchNorm3d(fout), torch.nn.ReLU())

    def forward(self, x):
        return self.block(x)

class Generator(torch.nn.Module):
    def __init__(self, seed_size, capacity=128):
        super(Generator, self).__init__()
        self.capacity = capacity

        self.embed = torch.nn.Linear(seed_size, capacity*3*16*16, bias=False)
        self.embedl = torch.nn.Linear(6, capacity*1*16*16, bias=False)

        self.resnet = torch.nn.ModuleList()
        for i in range(9): self.resnet.append(Block(capacity))
        self.resnet.append(Upsample(capacity, capacity, 4))

        self.image = torch.nn.Conv3d(capacity, 1, 3, padding=1, bias=True)
        self.bias = torch.nn.Parameter(torch.Tensor(1,3,64,64))

        for name, parm in self.named_parameters():
            if name.endswith('weight'): torch.nn.init.normal_(parm, 0, .05)
            if name.endswith('bias'): torch.nn.init.constant_(parm, 0.0)

    def forward(self, s,l):
        zx = torch.cat((self.embed(s).view(-1,self.capacity,3,16,16),
                        self.embedl(l).view(-1,self.capacity,1,16,16)),2)
        zx = F.relu(zx)
        for layer in self.resnet: zx = layer(zx)
        return torch.squeeze(torch.sigmoid(self.image(zx) + self.bias[None,:,:,:,:]))

class Discriminator(torch.nn.Module):
    def __init__(self, capacity=128, weight_scale=.01):
        super(Discriminator, self).__init__()
        self.capacity = capacity

        self.embed = torch.nn.Conv3d(1, capacity, 3, padding=1, bias=False)
        self.embedl = torch.nn.Linear(6, capacity*1*64*64, bias=False)

        self.resnet = torch.nn.ModuleList()
        self.resnet.append(DBlock(capacity, stride=4))
        for i in range(3): self.resnet.append(DBlock(capacity))

        self.out = torch.nn.Linear(capacity, 1, bias=True)

        for name, parm in self.named_parameters():
            if name.endswith('weight'): torch.nn.init.normal_(parm, 0, .05)
            if name.endswith('bias'): torch.nn.init.constant_(parm, 0.0)

    def forward(self, x,l):
        zx = torch.cat((self.embed(x[:,None,:,:,:]),
                        self.embedl(l).view(-1,128,1,64,64)),2)
        zx = F.relu(zx)
        for layer in self.resnet: zx = layer(zx)
        return self.out(zx.sum(dim=(2,3,4)))

def gradient_penalty(f, real_data, fake_data,l):
    alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size()).cuda()

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).detach()
    interpolates.requires_grad = True

    disc_interpolates = f(interpolates,l)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    return ((gradients.reshape(real_data.shape[0],-1).norm(2, dim=1) - 1) ** 2).mean()

##########################
### Fit Models 
##########################
seed_size = 128
batch_size = 32
dataset = SGZ(imgstat)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)

g = Generator(seed_size).cuda()
f = Discriminator().cuda()


fig, ax = plt.subplots(1,2,figsize=(20,10))

lr = 3e-4

foptimizer = torch.optim.Adam(f.parameters(), lr=lr, betas=(0,0.9))
goptimizer = torch.optim.Adam(g.parameters(), lr=lr, betas=(0,0.9))

losses = []

i = 0
t0 = time()
scores = []
epochs = 800
for epoch in range(epochs):        
    for x,l in dataloader:
        x = x.cuda()
        l = l.float().cuda()

        fake = g(torch.randn(x.shape[0], seed_size).cuda(),l)
        lossG = torch.mean(f(fake,l))

        # update G
        goptimizer.zero_grad()
        lossG.backward()
        goptimizer.step()

        # update D
        lossD = torch.mean(f(x,l))-torch.mean(f(fake.detach(),l))
        lossD = lossD + 10*gradient_penalty(f,x,fake.detach(),l) #lambda=10
        foptimizer.zero_grad()
        lossD.backward()
        foptimizer.step()
        

        losses.append(lossG.detach().cpu().numpy())
        i += 1

torch.save(g.state_dict(), "/content/drive/My Drive/CSE599iGM/Project/gpicklewcganFINAL")
torch.save(f.state_dict(), "/content/drive/My Drive/CSE599iGM/Project/fpicklewcganFINAL")



