import os
import random
from sklearn import manifold, datasets
from PIL import Image
import cosFace.faceNet as faceNet
import numpy as np
from cosFace.matlab_cp2tform import get_similarity_transform_for_PIL
import torch
from torch.autograd import Variable
import code


landMarkFile = './cosFace/data/casia_landmark.txt'
model_path = '/datasets/home/96/396/jbk001/cse152b_hw2release/cosFace/checkpoint_bn/checkpoint/netFinal_8.pth'
dataset_location = '/datasets/cse152-252-sp20-public/hw2_data/CASIA-WebFace'

def alignment(img, landmark ):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    
    ref_pts = np.array(ref_pts, dtype = np.float32 ).reshape(5, 2)
    landmark = np.array(landmark).astype(np.float32).reshape(5, 2)
    
    tfm = get_similarity_transform_for_PIL(landmark, ref_pts)
    
    img = img.transform(crop_size, Image.AFFINE,
            tfm.reshape(6), resample=Image.BILINEAR)
    
    img = np.asarray(img )
    if len(img.shape ) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    else:
        img = img[:, :, ::-1]

    img = np.transpose(img, [2, 0, 1] )
    return img


#get all the ids
ids = []
for idx in os.listdir(dataset_location):
    ids.append(idx)

#get random 10 ids
random_10 = random.sample(ids, 10)

#get the images for each path
random_10_images = {}
for idx, random_id in enumerate(random_10):
    random_10_images[random_id] = []
    for face in os.listdir(f'{dataset_location}/{random_id}'):
        random_10_images[random_id].append(Image.open(f'{dataset_location}/{random_id}/{face}'))
    
#Align all the faces
landmark = {}
with open(landMarkFile) as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0].split('/')[0]] = [float(k) for k in l[2:] if len(k) > 0]
for identity in random_10_images:
    print(f'aligning identity : {identity}')
    for face_idx, face in enumerate(random_10_images[identity]):
        random_10_images[identity][face_idx] = (alignment(face, landmark[identity] ).astype(np.float32 ) - 127.5) / 128
    


#get the model
model = faceNet.faceNet_BN()
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()
model.feature = True

#get the embedding of each random id
random_10_embeddings = []
class_partitions = []
count = 1
for identity in (random_10_images):
    for face in (random_10_images[identity]):
        count += 1
        #I KNOW THIS LINE OF CODE SUCKS, BUT DEAL WITH IT, IT'S 3AM RIGHT NOW BC I PLAYED TOO MUCH FIFA LAST NIGHT
        random_10_embeddings.append(model(Variable(torch.from_numpy(face.transpose(2, 0, 1).reshape((1,3,112,96))).float() ).cuda()).cpu().data.numpy())
    class_partitions.append(count)
    

embeddings = np.array(random_10_embeddings).squeeze()
np.save('embeddings.npy', embeddings)
np.save('classes.npy', class_partitions)
