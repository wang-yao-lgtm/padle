# 测试模型实例效果，读取部分数据输入模型

import argparse
import numpy as np
import repackage
repackage.up()
import collections
from mnists.dataloader import get_dataloaders
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from avalanche.models import SimpleMLP,SimpleCNN,MTSimpleCNN

#from mnists.models.classifier import CNN,CNN1


import collections

class simpleCNN_T(SimpleCNN):
    def __init__(self,num_classes=10):
        super(simpleCNN_T, self).__init__()
        self.num_classes=num_classes
        self.classifier = nn.Sequential (
            nn.Linear ( 64, self.num_classes ),
            )
        self.out=nn.LogSoftmax ( dim=1 )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.out(x)
        return x


# load data
def minis_load(count,data_label=[0,1,2,3,4,5,6,7,8,9]):

    dl_train,dl_test =get_dataloaders()
    train_val = []
    train_lab = []
    for k, data in enumerate(dl_train) :
        #print(np.shape(k))
        #plt.imshow(np.transpose(k, (1, 2, 0)))
        for ii in data_label:
            if int(data['labels'][0])==ii:
                train_val.append (data['ims'].reshape(3,32,32) )
                train_lab.append ( data['labels'] )
        if len(train_val) == count :
            break


    # load data
    #print ('xxx', type(train_val),np.array(train_val)[1,1:10,0:5 ] )
    train_X = train_val
    train_Y = train_lab
    # 可视化样本，下面是输出了训练集中前4个样本
    fig, ax = plt.subplots ( nrows=1, ncols=count, sharex='all', sharey='all' )
    ax = ax.flatten ()
    for i in range ( count ) :
        #train_X[i] = train_X[i].reshape(32, 32)

        #img = np.rollaxis ( train_X[i], 0, 3 )
        img=np.transpose ( train_X[i], (1, 2, 0) )
        # ax[i].imshow(img,cmap='Greys')
        ax[i].imshow ( img )
    ax[0].set_xticks ( [] )
    ax[0].set_yticks ( [] )
    plt.tight_layout ()
    plt.show ()
    return train_val,train_lab

# 加载数据
x,y=minis_load(5,data_label=[0,1,2,3,4,5,6,7,8,9])
print('targt',y)
def plot_x(x):
    fig, ax = plt.subplots ( nrows=1, ncols=1, sharex='all', sharey='all' )
    ax = ax.flatten ()
    img = np.transpose ( x, (1, 2, 0) )
    ax[0].imshow ( img )
    ax[0].set_xticks ( [] )
    ax[0].set_yticks ( [] )
    plt.tight_layout ()
    plt.show ()


#模型测试
from avalanche.models import SimpleMLP,SimpleCNN
#a=torch.load(f'models_clearn_BC2.pth').keys()
#print(a)
from avalanche.models import as_multitask
#model=SimpleCNN()
model=simpleCNN_T()
#model = as_multitask(cla, 'classifier')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model= torch.nn.DataParallel(model, device_ids=[0])

from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict =torch.load(f'models_clearn_mt_simplcnn40_overfit.pth') #预训练模型路径
#state_dict=torch.load(f'models_clearn_BC3.pth')
for k, v in state_dict.items():
    # 手动添加“module.”
    if 'model' in k:
        k = str(k).replace('model.features','features')
        #k = str ( k ).replace ( 'model.', '' ,1)
    elif str(k)=="classifier.classifiers.0.classifier.weight":
        k="classifier.0.weight"
    elif str(k)=="classifier.classifiers.0.classifier.bias":
        k="classifier.0.bias"
    else:
        continue
    new_state_dict[k]=v

model.load_state_dict(new_state_dict)



a=[]
for i in x:
    a.append(i.tolist())
print(model)
y_pre=model(torch.tensor(a))
print('-----------------------------------',y_pre)

#cla.load_state_dict(torch.load(f'models_clearn_simplcnn.pth'))


