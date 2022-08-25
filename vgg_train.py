import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
# from load_cifar10 import train_loader,test_loader
import os

from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
import glob


#load_cifar10

label_name = ["airplane","aotomobile","bird","cat","deer","dog","frog","horse","ship","truck"]   #类别列表
label_dict = {}

for idx,name in enumerate(label_name):     #类别字典
  label_dict[name] = idx

def default_loader(path):
  return Image.open(path).convert("RGB")      打开图像文件并转换成RGB

train_transform = transforms.Compose([     #￥数据增强
    transforms.RandomResizedCrop((28,28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomGrayscale(0.1),
    transforms.ColorJitter(0.3,0.3,0.3,0.3),
    transforms.ToTensor(),

])


class MyDataset(Dataset):                      #加载数据集
  def __init__(self,im_list,transform=None,loader=default_loader):
      super(MyDataset,self).__init__()
      imgs = []
      for im_item in im_list:
        im_label_name = im_item.split("/")[-2]
        imgs.append([im_item,label_dict[im_label_name]])

      self.imgs = imgs
      self.transform = transform
      self.loader = loader



  def __getitem__(self,index):
      im_path,im_label = self.imgs[index]
      im_data = self.loader(im_path)

      if self.transform is not None:
        im_data = self.transform(im_data)

      return im_data,im_label

  def __len__(self):
    return len(self.imgs)


im_train_list = glob.glob("/content/drive/MyDrive/datasets/cifar10/train/*/*.png")
im_test_list = glob.glob("/content/drive/MyDrive/datasets/cifar10/test/*/*.png")

train_dataset = MyDataset(im_train_list,transform = train_transform)
test_dataset = MyDataset(im_test_list,transform = transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,batch_size=6,shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=test_dataset,batch_size=6,shuffle=False,num_workers=2)





# VGG_define

class VGGbase(nn.Module):
  def __init__(self):
    super(VGGbase,self).__init__()


#  28*28
    self.covn1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.max_pooling1 = nn.MaxPool2d(kernel_size=2,stride=2)

#  14*14
    self.covn2_1 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    self.covn2_2 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    self.max_pooling2 = nn.MaxPool2d(kernel_size=2,stride=2)

#  7*7
    self.covn3_1 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )

    self.covn3_2 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )

    #如果不加padding尺寸会变成3*3
    self.max_pooling3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)


#  输入：4*4
    self.covn4_1 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    self.covn4_2 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.max_pooling4 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)

#  batchsize * 512 * 2 * 2 --> batchsize * (512 * 4)

    self.fc = nn.Linear(512*2*2,10)



  def forward(self,x):
    batchsize = x.size(0)
    out = self.conv1(x)
    out = self.max_pooling1(out)

    out = self.conv2_1(out)
    out = self.conv2_2(out)
    out = self.max_pooling2(out)

    out = self.conv3_1(out)
    out = self.conv3_2(out)
    out = self.max_pooling3(out)

    out = self.conv4_1(out)
    out = self.conv4_2(out)
    out = self.max_pooling4(out)

    out = out.view(batchsize,-1)
    out = self.fc(out)
    out = f.log_softmax(out,dim=1)

    return out

def VGGNet():
  return VGGbase()




# train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    #设置参数存储位置

epoch_num = 100
lr = 0.01

net = VGGNet().to(device)   #将网络参数存储到gpu

#loss
loss_func = nn. CrossEntropyLoss()   #使用交叉熵损失

#optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)   #每训练5轮更新一次lr，调整为上一次的0.9倍

for epoch in range(epoch_num):
  print("epoch:",epoch)
  net.train()

  for i,data in enumerate(train_loader):
    print("step",i)
    inputs,labels = data 
    inputs,labels = inputs.to(device),labels.to(device)

    output = net(inputs)
    loss = loss_func(outputs,labels)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


    _, pred = torch.max(outputs,dim=1)
    correct = pred.eq(labels.data).cpu().sum()
    print("step",i, "loss is :",loss.item(),"mini_batch correct is",100.0 * correct / batch_size)

  torch.save(net.state_dict(),"/content/drive/MyDrive/vgg/models/{}.pth".format(epoch+1))
  scheduler.step()
  print("lr is", optimizer.state_dict()["param_groups"][0]["lr"])