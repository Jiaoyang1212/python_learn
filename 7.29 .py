#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:14:01 2021

@author: jiaoyang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sample_size=11
x_array=np.linspace(0,10,sample_size)
slope=2
intercept=1
y_array=x_array*slope+intercept

std=1
epsilon=np.random.normal(0,std,sample_size)

y_array_2=y_array+epsilon
plt.figure()
plt.scatter(x_array,y_array)
plt.figure()
plt.scatter(x_array,y_array_2)

model=LinearRegression() #回归
model.fit(x_array.reshape(sample_size,1),y_array_2) #拟合，找系数
a=model.coef_
b=model.intercept_

z_array=np.linspace(0,1,100)
z_predict=model.predict(z_array.reshape(100,1)) #代入
plt.figure()
plt.scatter(z_array,z_predict)

x=np.zeros(100)
x_1=x.reshape(1,100) #把行向量变成列向量,括号里是（行数，列数）

from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
model_iris=LinearRegression()
model_iris.fit(x,y)
z=np.array([[5.6, 3, 6,2],
            [4.6, 3.1, 2, 0.5]])
z_predict=model_iris.predict(z)
print(model.score(x,y))

#%% 离散
def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y

x=np.linspace(-5,5,101) #从-5到5取101个点
y=sigmoid(x)
plt.plot(x,y)

#%%
sample_size=100 #在平面上生成100个点
X=np.random.uniform(0,10,(sample_size,2))   #生成100个随机的二维点
x1=X[:,0] #冒号前是第一个坐标，后是第二个坐标
x2=X[:,1]
plt.scatter(x1,x2)
y=np.zeros(sample_size)

z=np.random.normal(0,0.5,sample_size) #增加噪音
for i in range(sample_size):
    if x2[i]>x1[i]+z[i]:
        y[i]=1
    else:
        y[i]=0

y1_index=(y==1) #两个等号是判断\
y0_index=(y==0)

#plt.scatter(x1[y0_index],x2[y0_index],color='blue')
#plt.scatter(x1[y1_index],x2[y1_index],color='red')

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X,y)
print(model.intercept_)
print(model.coef_)
model.fit(X,y)#找系数

X_test=np.array([
    [10,0],
    [0,10],
    [5,2]
    ])
y_test_pred=model.predict(X_test)

xt1=np.arange(-0.1,10.1,0.1)
xt2=np.arange(-0.1,10.1,0.1)
xxt1,xxt2=np.meshgrid(xt1,xt2) #xxt1每行的每个元素从-0.1到10 xx2每列


xt=np.hstack([xxt1.reshape(-1,1), xxt2.reshape(-1,1)])
yt=model.predict(xt)

plt.scatter(xt[yt==0,0],xt[yt==0,1],color="b")#yt都是0，第一列都是0,...
plt.scatter(xt[yt==1,0],xt[yt==1,1],color="r")


#%% 7.30
a=[1,2,3] #list 可加减

a=[] #空的list
a.append(1) #添加元素
a.append(10)
a

a+a #没有相加，只是把list延长
100*a


b=np.array(a) #把list换成array
b
b+b
c=list(b) #把array换成list
c

a=np.ones(10) #10个1
b=a
a
b
b[0]=2 # 把b的第一个元素改成2
b
a #a也会变

#%% KNN模型(参数k= ，选周围多少个点) 选点过少过拟合，过多欠拟合
import pandas as pd #读取csv文件
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

path="/Users/jiaoyang/Desktop/未命名文件夹/iris.csv"
data=pd.read_csv(path,encoding='gbk',index_col=0) #有中文的时候用encoding
data.columns #显示列的名字
data.columns
data_v=data.values
#调某行数据
data.iloc[0,0]#第一行第一列
data.iloc[0,:]
data.iloc[1,:]
data.iloc[:,0]
data.iloc[:,-1]#行全取，只取最后一列

model=KNeighborsClassifier(n_neighbors=5,p=1) #找一个最近点
X=data.iloc[:,0:4] #取所有行和三列 第一列到第4列
y=data.iloc[:,4]

model.fit(X,y)
model.score(X,y)

area1=data.iloc[:,0]*data.iloc[:,1]
area2=data.iloc[:,2]*data.iloc[:,3] #第三列乘第四列
data['area1']=area1 #加一列area1
data['area2']=area2
data_final=data.iloc[:,[0,1,2,3,5,6,4]]
save_path=path="/Users/jiaoyang/Desktop/未命名文件夹/iris_new.csv"
data_final.to_csv(save_path)

#%%

X=data_final.iloc[:,:-1]
y=data_final.iloc[:,-1]

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.4)
model=KNeighborsClassifier(n_neighbors=1) 
model.fit(train_X,train_y)
print(model.score(train_X,train_y))#训练级
print(model.score(test_X,test_y))#测试级

#%% 优化 找极值
#xn=xn-1 - a f'(xn-1) a为任意系数
def f(x):
    return x*x -2*x-5

def fderiv(x):
    return 2*x-2
learning_rate=0.1
n_iter=100
xs=np.zeros(n_iter+1)
xs[0]=100

for i in range(n_iter):
    xs[i+1]=xs[i]-learning_rate*fderiv(xs[i])
plt.plot(xs)

#%%
from scipy.optimize import minimize #sin函数
a=minimize(f,x0=100).x

def f2(x):
    return np.exp(-x**2)*(x**2)

f2(1)
a=minimize(f2,x0=5).x
b=minimize(f2,x0=-5).x

#%% 决策树
def E(x):
    a=-x*np.log(x)-(1-x)*np.log(1-x)
    return a 
x=np.linspace(0.01,0.99,100)
y=E(x)
plt.figure()
plt.plot(x,y)

wm=pd.read_csv("/Users/jiaoyang/Desktop/未命名文件夹/wm_1.csv",encoding='gbk',index_col=0)
x=wm.iloc[:,:6]
y=wm.iloc[:,6]
import sklearn.tree as tree
model=tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=2)
model.fit(x,y) 
tree.plot_tree(model)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
bc=load_breast_cancer()
X=bc.data
y=bc.target
X_train,X__test,y_train,y_test=\
    train_test_split(X,y,train_size=0.6)
import sklearn.tree as tree
model=tree.DecisionTreeClassifier(max_depth=2)
model.fit(X_train,y_train)
tree.plot_tree(model)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

#用KNN 容易过拟合
model_1=KNeighborsClassifier(n_neighbors=20)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))


#%%画圆
sample_size=1000
x=np.random.uniform(-10,10,(sample_size,2)) #选数组
plt.scatter(x[:,0],x[:,1])
radius=7
label=np.zeros(sample_size)

for i in range(sample_size):
    if np.sqrt(x[i,0]**2+x[i,1]**2)<radius:
        label[i]=1

plt.figure(figsize=(10,10))
plt.scatter(x[label==0,0],x[label==0,1],color='yellow')
plt.scatter(x[label==1,0],x[label==1,1],color='pink')

#%%7.30作业
#1 对称阵
import numpy as np
n=5
mat=np.zeros((n,n))
mat[0,:]=np.arange(1,n+1,1)
for i in range(1,n):
    mat[i,:]=mat[i-1,0]
    for j in range(n-1):
        mat[i,j]=mat[i-1,j+1]
    
for i in range(n):
    for j in range(n):
        mat[i,j]=(i+j)%n+1
    
#2 n的阶乘
n=6
res=1 #定义结果
for i in range(1,n+1):
    res=res*i
print(res)

#3minimize一个函数
def f(x):
    return x**4-x**2
print(f(2))
solution=minimize(f,x0=1).x
print(solution)
plt.plot(f(x))

#%%补缺失值
import numpy as np
import padans as pd

data=np.random.randn(5,3)
df=pd.DataFrame(data,columns=['one','two','three'])
df.iloc[0,1]=np.nan            #给第一行第二列设置空值
df.iloc[1,2]=np.nan

#使用0填充空值
df.fillna(0,inplace=Ture)
df_1=df.fillna(0)

df_2=df.fillna(method='ffill')#延用上上一个时间的值
df_3=df_2.fillna(0)

#使用平均值进行填充
mean=df['three'].mean()
df['three'].fillna(mean,inplace=True)
#or df.iloc[:,1].fillna(mean,inplace=True)

#%% neueal network
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

sample_size=100

X=np.linspace(0,6,sample_size)
y=X**2
for i in range(1):


    size=(i+1)*100
    model=MLPRegressor(
       hidden_layer_sizes=(size, 200),
       activation='relu') #几个隐含层几个神经元 前是神经元
    model.fit(X.reshape((sample_size,1)),y)

    y_pre=model.predict(X.reshape((sample_size,1)))
    plt.show()
    plt.scatter(X,y)
    plt.scatter(X,y_pre)
   
 
#%%
from sklearn.linear_model import LogisticRegression


sample_size=1000
x=np.random.uniform(-10,10,(sample_size,2)) #选数组
plt.scatter(x[:,0],x[:,1])
radius=7
lable=np.zeros(sample_size)

for i in range(sample_size):
    if np.sqrt(x[i,0]**2+x[i,1]**2)<radius:
        lable[i]=1

plt.figure(figsize=(10,10))
plt.scatter(x[lable==0,0],x[lable==0,1],color='blue')
plt.scatter(x[lable==1,0],x[lable==1,1],color='pink')

model=LogisticRegression()
model. fit(x,lable)
model.score(x,lable)

x_test=np.random.uniform(-10,10,(10000,2))
lable_pred=model.predict(x_test)
  
plt.figure(figsize=(10,10))
plt.scatter(x_test[lable_pred==0,0],x_test[lable_pred==0,1],color='blue')
plt.scatter(x_test[lable_pred==1,0],x_test[lable_pred==1,1],color='pink')

#KNN
model=KNeighborsClassifier()
model. fit(x,lable)
model.score(x,lable)

x_test=np.random.uniform(-10,10,(10000,2))
lable_pred=model.predict(x_test)
  
plt.figure(figsize=(10,10))
plt.scatter(x_test[lable_pred==0,0],x_test[lable_pred==0,1],color='blue')
plt.scatter(x_test[lable_pred==1,0],x_test[lable_pred==1,1],color='pink')

#决策树
import sklearn.tree as tree
model=tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=1000000)

model. fit(x,lable)
plt.figure(figsize=(5,5))
x_test=np.random.uniform(-10,10,(10000,2))
lable_pred=model.predict(x_test)
  

plt.scatter(x_test[lable_pred==0,0],x_test[lable_pred==0,1],color='blue')
plt.scatter(x_test[lable_pred==1,0],x_test[lable_pred==1,1],color='pink')

#
from sklearn.neural_network import MLPClassifier

model=MLPClassifier(hidden_layer_sizes=(1000,10))
model. fit(x,lable)
model.score(x,lable)

x_test=np.random.uniform(-10,10,(10000,2))
lable_pred=model.predict(x_test)
  
plt.figure(figsize=(10,10))
plt.scatter(x_test[lable_pred==0,0],x_test[lable_pred==0,1],color='blue')
plt.scatter(x_test[lable_pred==1,0],x_test[lable_pred==1,1],color='pink')


#%%
    
    


