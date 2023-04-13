import torch
import os
import math
import random
from matplotlib import pyplot as pl

##拟合数据
def synthetic_data(w,b,num) :
    x = torch.normal(0,1,(num,len(w)))
    y = torch.matmul(x,w)+b
    y += torch.normal(0,0.01,(y.shape))
    return x,y.reshape(-1,1)
#抽取样本
def data_iter (batches_size,features,labels) :
    num = len(features)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(batches_size) :
        batch_indices = torch.tensor(
            indices[i:min(i+batches_size,num)]
        )
        yield features[batch_indices],labels[batch_indices]
#线性函数
def liner(x,w,b) :
    return torch.matmul(x,w)+b
#损失函数
def loss (y_hat,y) :
    return (y_hat-y.reshape(y_hat.shape))**2/2
#梯度下降
def gradient (params,lr,batches_size) :
    with torch.no_grad() :
        for param in params :
            param-=lr*param.grad/batches_size
            param.grad.zero_()


xaxis=[]
yaxis=[]
iteratornum=0
true_w = torch.tensor([2,-3.4])
true_b = 6
##生成的特征和标签
features,labels = synthetic_data(true_w,true_b,1000)
##学习率
lr = 0.03
##训练次数
train_num = 100
batches_size = 10
##初始化参数
w = torch.tensor([100.0,100.0],requires_grad=True)
b = torch.tensor([100.0],requires_grad=True)

for num in range(train_num) :
    for x,y in data_iter(batches_size,features,labels) :
        result = loss(liner(x,w,b),y)
        yaxis.append(result.sum().detach())
        xaxis.append(iteratornum)
        iteratornum+=1
        result.sum().backward()
        gradient([w,b],lr,batches_size)
print(f"w={w},b={b}")

##损失函数值对训练次数的变化
pl.plot(xaxis,[float(i) for i in yaxis])
pl.show()




