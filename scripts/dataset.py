"""dataset.py"""

import os
import random
import numpy as np

import torch
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class WholeDataLoader(Dataset):
    def __init__(self,train):
   
        #self.indices = range(len(self))
        if train==0:
            self.img_list=os.listdir('./data/CelebA/train/train')
        elif train==1:
            self.img_list=os.listdir('./data/CelebA/val/val')
        else :
            self.img_list=os.listdir('./data/CelebA/test/test')


        #import pdb;pdb.set_trace()
        self.img_list.sort()
        self.transform=transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor(),])
        self.att=[]
        self.train=train
        with open('./data/CelebA/list_attr_celeba.csv','r') as f:
            reader=csv.reader(f)
            att_list=list(reader)
        att_list=att_list[1:]
        with open('./data/CelebA/list_eval_partition.csv','r') as f:
            reader=csv.reader(f)
            eval_list=list(reader)
    
        for i,eval_inst in enumerate(eval_list):
            
            if eval_inst[1]==str(self.train):
                if att_list[i][0]==eval_inst[0]:
                    self.att.append(att_list[i])
                else:
                    pass

        
        self.att=np.array(self.att)
        
        self.att=(self.att=='1').astype(int)
        
        
       
       


    def __getitem__(self, index1):
        
        heavy_makeup=self.att[index1][19]
        male=self.att[index1][21]
        Chubby=self.att[index1][14]
        eye=self.att[index1][16]
        wavy=self.att[index1][34]
        blond=self.att[index1][10]
        mustache=self.att[index1][23]
        black=self.att[index1][9]
        nobeard=self.att[index1][25]
        young=self.att[index1][40]
        attractive=self.att[index1][3]
    
        high_cheekbone=self.att[index1][20]
        bald=self.att[index1][5]

        smile=self.att[index1][32]
        check=self.att[index1][5]
        bignose=self.att[index1][8]
        slight=self.att[index1][22]
        lipstick=self.att[index1][37]
    
        index2=random.choice(range(len(self.img_list)))
        if self.train==0:
            img1=Image.open('./data/CelebA/train/train/'+self.img_list[index1])
            img2=Image.open('./data/CelebA/train/train/'+self.img_list[index2])
        elif self.train==1:
            img1=Image.open('./data/CelebA/val/val/'+self.img_list[index1])
            img2=Image.open('./data/CelebA/val/val/'+self.img_list[index2])
        else:
            img1=Image.open('./data/CelebA/test/test/'+self.img_list[index1])
            img2=Image.open('./data/CelebA/test/test/'+self.img_list[index2])
    
     
        return self.transform(img1), self.transform(img2),attractive,male
        


    def __len__(self):
        return len(self.att)







def return_data(args,train):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_workers = args.num_workers
    image_size = args.image_size

   
    assert image_size == 64, 'currently only image size of 64 is supported'

    dset = WholeDataLoader(train)
    
    if train==0:
        train_loader = DataLoader(dset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)
        data_loader=train_loader
    elif train==1:
        eval_loader = DataLoader(dset,
                                batch_size=eval_batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)
        data_loader=eval_loader
    else :
        test_loader = DataLoader(dset,
                                batch_size=eval_batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)
        data_loader=test_loader
    
    return data_loader

