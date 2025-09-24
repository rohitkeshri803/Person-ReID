 
import torch.utils.data as data
import torch
import os
import random
from torchvision import transforms
import numpy as np
from PIL import Image


DATASET_BASE = os.environ.get("MARKET1501_DIR", "/path/to/Market-1501-v15.09.15")
IMG_SIZE = 256
CROP_SIZE = 224
BATCHSIZE = 4
IMG_EXT = 'jpg'


def load_model(path=None):
    if not path:
        return None
    full = os.path.join(DATASET_BASE, 'models', path)
    for i in [path, full]:
        if os.path.isfile(i):
            return torch.load(i)
    return None
    
    
class Market1501(data.Dataset):
    
    def __init__(self):
        
        self.Batchsize = BATCHSIZE
        self.Type = 'train'
        self.transform = transforms.Compose([
        #transforms.Scale(IMG_SIZE),
        transforms.Resize(IMG_SIZE),
        #transforms.RandomSizedCrop(CROP_SIZE),
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        #transforms.Scale(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.Type == 'train':

            self.Image_dir = os.path.join(DATASET_BASE, 'bounding_box_train')
            self.imagepaths = [os.path.join(root, file)  for root, dirs, files in os.walk(self.Image_dir)
                        for file in files if IMG_EXT in file]
            self.N = len(self.imagepaths)
            
            self.labels = [i.split('/')[-1].split('_')[0] for i in self.imagepaths]
            indexdict = {i:idx for idx,i in enumerate(np.unique(self.labels))}
            self.labels = [indexdict[i] for i in self.labels]
            random_shuffle = np.random.choice(self.N,self.N,replace = False)  
            
            self.imagepaths = np.array(self.imagepaths)[random_shuffle]
            self.labels = np.array(self.labels)[random_shuffle]
            
            self.dict = self.get_path_dict(self.imagepaths, self.labels)

        else:
            self.Image_dir = os.path.join(DATASET_BASE, 'bounding_box_test')
            self.query_dir = os.path.join(DATASET_BASE, 'query')
            self.imagepaths = [os.path.join(root, file)  for root, dirs, files in os.walk(self.Image_dir)
                        for file in files if IMG_EXT in file]
            #print (len(self.imagepaths))
            self.imagepaths = [i for i in self.imagepaths if ('0000' not in i.split('/')[-1]) and ('-1' not in i.split('/')[-1])]  
            self.querypaths = [os.path.join(root, file)  for root, dirs, files in os.walk(self.query_dir)
                        for file in files if IMG_EXT in file]
            #print (len(self.imagepaths))
            #self.imagepaths = self.imagepaths + self.querypaths
            self.querylabels = [i.split('/')[-1].split('_')[0] for i in self.imagepaths]
            queryindexdict = {i:idx for idx,i in enumerate(np.unique(self.querylabels))}
            self.querylabels = [queryindexdict[i] for i in self.querylabels]
            self.N = len(self.imagepaths)
            self.labels = [i.split('/')[-1].split('_')[0] for i in self.imagepaths]
            indexdict = {i:idx for idx,i in enumerate(np.unique(self.labels))}
            self.labels = [indexdict[i] for i in self.labels]
            random_shuffle = np.random.choice(self.N,self.N,replace = False)  
            
            self.imagepaths = np.array(self.imagepaths)[random_shuffle]
            self.labels = np.array(self.labels)[random_shuffle]
            
            self.dict = self.get_path_dict(self.imagepaths, self.labels)
            #print (len(self.imagepaths))

       
        
    def _len_(self):
        
        return self.N 

    def testlen(self):
        return self.N, len(self.querypaths)
      
    def get_path_dict(self, impaths, labels):

        d = {i:[] for i in labels}
  
        for i,j in zip(impaths,labels):
            d[j].append(i)

        return d
        
    def sample_random_indices(self, n, labelled_Bucket_index):

        if labelled_Bucket_index != []:
            exclude = np.array(labelled_Bucket_index)
            def rand(n, exclude):
                r = None
                while r in exclude or r is None:
                    r = np.random.choice(self.N)
                return r
            tmp = []
            exclude_list = []
            for i in range(n):
                var = rand(n, exclude_list)
                tmp.append(var)
                exclude_list.extend(np.unique((exclude[np.where(np.sum(exclude==[var],1)>0)])))
                print (exclude_list)
            return tmp
        else:
            return np.random.choice(self.N,n, replace = False)

    def process_img(self, img_path):

        img_full_path = os.path.join(self.Image_dir, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.Type == 'train':
            img = self.transform(img)
        elif self.Type == 'test':
            img = self.transform_test(img)
        return np.array(img, 'float64')

    def nextbatch(self, n, exclude_labelled_Bucket = []):
        
        # imgs = []
        # labels = []
        # query = []
        
        random_sample_indices = self.sample_random_indices(n, exclude_labelled_Bucket)
        random_sample_index_for_query = self.sample_random_indices(1, exclude_labelled_Bucket)
        
        imgs = np.stack([self.process_img(impath) for impath in self.imagepaths[random_sample_indices]], 0)
        img_labels = self.labels[random_sample_indices]
        query = np.stack([self.process_img(impath) for impath in self.imagepaths[random_sample_index_for_query]], 0)
        query_label = self.labels[random_sample_index_for_query]

        # print(imgs.shape)             #(30, 3, 224, 224)
        # print(img_labels.shape)       #(30,)
        # print(query.shape)            #(1, 3, 224, 224)
        # print(query_label.shape)      #(1,)

        # imgs.append(imgs)
        # labels.append(img_labels)
        # query.append(query)
        return torch.Tensor(query), torch.Tensor(query_label), torch.Tensor(imgs), torch.Tensor(img_labels)