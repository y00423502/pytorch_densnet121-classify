# 导入需要的模块

import sys
sys.path.append("./lib")
import getopt #在外部使用参数传递

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
parser.add_argument('--p', type=str, default='./output_models/extract_EmbedingFeature/czx_fire_30k/')  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--re', type=str, default='./output_models/normal/czx_fire_30k/best_500epoch.pth')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--eps', type=int, default=20)
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
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



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

#data_dir = './data/czx_fire_30k'
data_dir = opt.d
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.b,
                                             shuffle=True, num_workers=opt.num)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

# Imshow Image

#可自选打开
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
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# Define Train_Function

def train_model(model, criterion, optimizer, scheduler, num_epochs=500):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs,feature = model(inputs)    #如果是normal模式，打开这句，注释下面一句
#                     print(outputs.shape)
#                     print(feature.shape)
                    #outputs,embedding_tensor = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Training loop complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #这里是选择每间隔10epoch保存一个模型，这个可以根据模型训练时间选择
        if epoch>0 and epoch % (opt.eps) == 0:
            #save_path='./output_models/extract_EmbedingFeature/czx_fire_30k/'            #保存模型的路径根据需要修改
            save_path=opt.p  
            torch.save(model.state_dict(),save_path+'%d.pth' % (epoch))
            print('save done')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model                          #返回500epoch中最佳的模型

# 可视化验证分类结果 （可选）

#  def visualize_model(model, num_images=20):    #这里的20可以根据需要更改展示的图片数
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)

# Define Network && Load Pretrained Models

#Pretrained_model_path = "../models/densenet121_fire_ep15.pth"   这里可以选择是否载入预训练模型，我这里是从头训练，注释掉

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#如果是单gpu训练这里的cuda号对应使用哪块gpu训练

#model = models.densenet121(pretrained=False)
model = extract_EmbedingFeature_densenet121.densenet121(pretrained=False)  #这里如果是需要抽特征的用这句，test也是
#如果是采用官方原版的densenet这里就需要把下面两行打开，更改分类头
# num_ftrs = model.classifier.in_features
# #更换分类头从1000类转到2类
# model.classifier = nn.Linear(num_ftrs, 2)
print(model) #可以打开查看网络结构

#一般pytorch默认是单gpu训练，下面我们采用多gpu并行训练加快训练速度
if torch.cuda.device_count() > 1:
    #这里是训练加载多gpu的操作,默认是所有的gpu都用上
    model = torch.nn.DataParallel(model) 
    #model = torch.nn.DataParallel(model, device_ids=[0,1])     #这里可以指定选择0，1两块gpu                                 
    model = model.to(device)
    #model_ft.load_state_dict(torch.load(Pretrained_model_path))  #如需要载入预训练模型加快训练速度，打开这句
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--d', type=str, default='./data/czx_fire_30k')  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
#     parser.add_argument('--p', type=str, default='./output_models/extract_EmbedingFeature/czx_fire_30k/')  # effective bs = batch_size * accumulate = 16 * 4 = 64
#     parser.add_argument('--re', type=str, default='./output_models/normal/czx_fire_30k/best_500epoch.pth')
#     opt = parser.parse_args()
    #output_model_SavePath='./output_models/normal/czx_fire_30k/best_500epoch.pth'
output_model_SavePath=opt.re
output_model= train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=opt.epochs)

#visualize_model(output_model) 可选

torch.save(model_ft.state_dict(), output_model_SavePath)

