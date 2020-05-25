import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import faceNet
import torch.nn as nn
import os
import numpy as np

from tensorboardX import SummaryWriter
import datetime
import random
import shutil

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--imageRoot', default='/datasets/cse152-252-sp20-public/hw2_data/CASIA-WebFace', help='path to input images')
parser.add_argument('--alignmentRoot', default='./data/casia_landmark.txt', help='path to the alignment file')
parser.add_argument('--experiment', default='checkpoint', help='the path to store sampled images and models')
parser.add_argument('--marginFactor', type=float, default=0.35, help='margin factor')
parser.add_argument('--scaleFactor', type=float, default=30, help='scale factor')
parser.add_argument('--imHeight', type=int, default=112, help='height of input image')
parser.add_argument('--imWidth', type=int, default=96, help='width of input image')
parser.add_argument('--batchSize', type=int, default=128, help='the size of a batch')
parser.add_argument('--nepoch', type=int, default=20, help='the training epoch')
parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
parser.add_argument('--iterationDecreaseLR', type=int, nargs='+', default=[16000, 24000], help='the iteration to decrease learning rate')
parser.add_argument('--iterationEnd', type=int, default=28000, help='the iteration to end training')

# The detail network setting
opt = parser.parse_args()

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



# Initialize network
net = faceNet.faceNet_BN(m = opt.marginFactor, feature = True )
state_dict = torch.load('/datasets/home/96/396/jbk001/cse152b_hw2-release/cosFace/checkpoint_bn/checkpoint/netFinal_8.pth')
net.load_state_dict(state_dict)

# Move network and containers to gpu
if not opt.noCuda:
    net = net.cuda(opt.gpuId )

    
    
#get the random 10 
if os.path.isdir('./random_10'):
    shutil.rmtree('./random_10')
os.mkdir('./random_10')
#get all the ids
ids = []
for idx in os.listdir(opt.imageRoot):
    ids.append(idx)
#get random 10 ids
random_10 = random.sample(ids, 10)
for random_identity in random_10:
    shutil.copytree(f'{opt.imageRoot}/{random_identity}', f'./random_10/{random_identity}')


# Initialize dataLoader
faceDataset = dataLoader.BatchLoader(
        imageRoot = './random_10',
        alignmentRoot = opt.alignmentRoot,
        cropSize = (opt.imWidth, opt.imHeight)
        )
faceLoader = DataLoader(faceDataset, batch_size = opt.batchSize, num_workers = 16, shuffle = False )
    
embeddings = []
targets = []
    
lossArr = []
accuracyArr = []
iteration = 0
for i, dataBatch in enumerate(faceLoader ):
    print(f'getting embeddings : {i}')

    iteration += 1

    # Read data
    image_cpu = dataBatch['img']
    imBatch = Variable(image_cpu )

    target_cpu = dataBatch['target']
    targetBatch = Variable(target_cpu )

    if not opt.noCuda:
        imBatch = imBatch.cuda()
        targetBatch = targetBatch.cuda()

   
    pred = net(imBatch ).cpu().data.numpy().squeeze()
    embeddings.extend(pred)
    targets.extend(np.array(target_cpu).squeeze())
        

embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)
np.save('targets.npy', targets)
        
        
    
    
    
   


