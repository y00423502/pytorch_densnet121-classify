# 导入需要的模块

import sys
sys.path.append("./lib")

import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import cv2
import random
import os
import numpy as np
import json
import time
import copy
from PIL import Image
from PIL import Image
import extract_EmbedingFeature_densenet121 #这里这个包是基于原生densenet修改的返回值是分类结果，EmbedingFeature特征
import sklearn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='./data/czx_fire_30k')  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
parser.add_argument('--p', type=str, default='../fire_RGB_classify_pro/300.pth')  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--re', type=str, default='./data/czx_fire_30k/result/')
parser.add_argument('--b', type=int, default=32)
parser.add_argument('--num', type=int, default=8)
opt = parser.parse_args()

# Data Pre_Process

'''
Define data_transforms，将训练及测试验证的图片裁剪到256
随机扩增并转换到tensor,(256,256,3)-->（3,256,256） 并将（0，255）->归一化到 (0,1)
transforms.Normalize做规范化，(0,1)-->(-1,1),加速收敛，防止梯度消失
'''

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Define反规范化 保存测试结果图像的函数

def transform_invert(img):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    img = img.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img = np.array(img) * 255

    if img.shape[2] == 3:
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
    elif img.shape[2] == 1:
        img = Image.fromarray(img.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img.shape[2]) )

    return img

# Load Data

'''
给出data路径
文件格式，czx_fire_30k路径下有三个文件夹
- train
------fire
------nofire
- val
------fire
------nofire
- test
------fire
------nofire
'''
x='test'
#data_dir = './data/czx_fire_30k'
data_dir = opt.d
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.b,
                                             shuffle=True, num_workers=opt.num)}
dataset_sizes = {x: len(image_datasets[x])}
class_names = image_datasets['test'].classes
testloader=dataloaders['test']
print(class_names)

# Imshow Image

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['test']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# Define Test Function

def test_model(model_path, testloader,TestResult_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    #model = extract_EmbedingFeature_densenet121.densenet121(pretrained=False) #这里如果是需要抽特征的用这句，test也是
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    #这里注意下，如果训练的时候用的多gpu必须加上这句，并且gpu号数对应，单gpu训练不需要
    model = torch.nn.DataParallel(model) 
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()                          #测试时必须加这句， 否则的话，有输入数据，即使不训练，它也会改变权值
    print(model)
    print('model load done')
    
    #下面开始测试
    with torch.no_grad():
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        index=0
        index_fire=0
        index_nofire=0
        for data in testloader:
            #if index==0:
            images_numpy, labels = data
            images, labels = images_numpy.to(device), labels.to(device)
            # 预测
            #outputs,featute = model(images)  #这里是用到embedding_feature_densenet时可以选择打开
            outputs = model(images)
            #print(featute.shape)
            # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
            pred= torch.max(outputs, 1)[1]
            try:
                for i in range(32):       #这里的 range 32跟batchsize要匹配
                    images=images_numpy[i]
                    for j in range(len(mean)): #反标准化保存
                        images[j] = images_numpy[i][j] * std[j] + mean[j]
                    images=transform_invert(images)
                    label=pred[i].cpu().detach().numpy()
                    index +=1
                    if label==1:
                        index_nofire+=1
                        result_nofire_path=TestResult_path+'nofire/'
                        if not os.path.exists(result_nofire_path):
                            os.makedirs(result_nofire_path)
                        images.save(result_nofire_path+str(index)+'.jpg')
                    else:
                        index_fire+=1
                        result_fire_path=TestResult_path+'fire/'
                        if not os.path.exists(result_fire_path):
                            os.makedirs(result_fire_path)
                        images.save(result_fire_path+str(index)+'.jpg')
            except:
                print('batchsize cannot reach 32')  #这里的32是前面我们设定的batchsize
                
    print('Test and Save result done\nTest %d images done\nFire: %d images\nNoFire: %d images'%(index,index_fire,index_nofire))

# Test


model_path=opt.p
TestResult_path=opt.re+x+'/'
#model_path='../fire_RGB_classify_pro/300.pth'
#TestResult_path='./data/czx_fire_30k/result/'+x+'/'
if not os.path.exists(TestResult_path):
    os.makedirs(TestResult_path)

test_model(model_path, testloader,TestResult_path)

