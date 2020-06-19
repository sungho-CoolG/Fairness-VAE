import os
import sys
import json
import random
import logging
import torch.utils.data as data
import torch
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
# from .transformer import get_transformer, load_image



#indices = range(len(self))
# if train==0:
img_list=os.listdir('./data/CelebA/train/train')
    # import pdb;pdb.set_trace()
# elif train==1:
#     img_list=os.listdir('./data/val/')
# else :
#     img_list=os.listdir('./data/test/')


#import pdb;pdb.set_trace()
img_list.sort()

# number of data
# [18177.0, 43277.0, 83602.0, 33280.0, 3713.0, 24685.0, 39213.0, 38341.0, 38906.0, 24267.0, 8362.0, 33192.0, 23386.0, 9389.0, 7571.0, 10521.0, 10337.0, 6896.0, 62554.0, 73645.0, 
# 68261.0, 78486.0, 6642.0, 18869.0, 135778.0, 46101.0, 7005.0, 44846.0, 13040.0, 10525.0, 9156.0, 78080.0, 33947.0, 51982.0, 30362.0, 8039.0, 76436.0, 19763.0, 11890.0, 126787.0]

# att_male :  [7053.0, 23240.0, 41280.0, 17821.0, 1787.0, 12263.0, 20738.0, 20379.0, 17979.0, 13815.0, 3673.0, 15762.0, 10349.0, 4970.0, 4805.0, 4994.0, 3878.0, 3421.0, 34230.0, 52431.0, 28915.0, 78486.0, 2307.0, 12045.0, 67989.0, 25638.0, 2363.0, 21623.0, 6813.0, 7810.0, 3076.0, 59378.0, 15886.0, 26634.0, 18754.0, 3963.0, 41067.0, 11684.0, 5191.0, 60763.0]
# att_female :  [11124.0, 20037.0, 42322.0, 15459.0, 1926.0, 12422.0, 18475.0, 17962.0, 20927.0, 10452.0, 4689.0, 17430.0, 13037.0, 4419.0, 2766.0, 5527.0, 6459.0, 3475.0, 28324.0, 21214.0, 39346.0, 0.0, 4335.0, 6824.0, 67789.0, 20463.0, 4642.0, 23223.0, 6227.0, 2715.0, 6080.0, 18702.0, 18061.0, 25348.0, 11608.0, 4076.0, 35369.0, 8079.0, 6699.0, 66024.0]
# no_att_male :  [71433.0, 55246.0, 37206.0, 60665.0, 76699.0, 66223.0, 57748.0, 58107.0, 60507.0, 64671.0, 74813.0, 62724.0, 68137.0, 73516.0, 73681.0, 73492.0, 74608.0, 75065.0, 44256.0, 26055.0, 49571.0, 0.0, 76179.0, 66441.0, 10497.0, 52848.0, 76123.0, 56863.0, 71673.0, 70676.0, 75410.0, 19108.0, 62600.0, 51852.0, 59732.0, 74523.0, 37419.0, 66802.0, 73295.0, 17723.0]
# no_att_female :  [73159.0, 64246.0, 41961.0, 68824.0, 82357.0, 71861.0, 65808.0, 66321.0, 63356.0, 73831.0, 79594.0, 66853.0, 71246.0, 79864.0, 81517.0, 78756.0, 77824.0, 80808.0, 55959.0, 63069.0, 44937.0, 84283.0, 79948.0, 77459.0, 16494.0, 63820.0, 79641.0, 61060.0, 78056.0, 81568.0, 78203.0, 65581.0, 66222.0, 58935.0, 72675.0, 80207.0, 48914.0, 76204.0, 77584.0, 18259.0]
 

with open('./data/CelebA/list_attr_celeba.csv','r') as f:
    reader=csv.reader(f)
    att_list=list(reader)

att=list(0. for i in range(41))
att_male =list(0. for i in range(41))
no_att_male =list(0. for i in range(41))
att_female =list(0. for i in range(41))
no_att_female =list(0. for i in range(41))


for i in range(1, len(img_list)):
    # for j in range(40):
    for j in range(1,41):
    
        if (att_list[i][j] == '1') and (att_list[i][21] == '1'):
        #  and att_list[i][1:][j]
            att_male[j] += 1
        elif (att_list[i][j] == '-1') and (att_list[i][21] == '1'):
        #  and att_list[i][1:][j]
            no_att_male[j] += 1
        elif (att_list[i][j] == '1') and (att_list[i][21] == '-1'):
            att_female[j] += 1
        else :
            no_att_female[j] += 1

print('att_male : ',att_male)
print('att_female : ',att_female)
print('no_att_male : ',no_att_male)
print('no_att_female : ',no_att_female)


# df = pd.DataFrame.from_records()
# import pdb;pdb.set_trace()
# att=(att=='1').astype(int)
# import pdb;pdb.set_trace()

print('123123')