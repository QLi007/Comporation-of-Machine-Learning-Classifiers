# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:16:42 2020

@author: lijj
"""
#hw3 q5
import numpy as np
import csv
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



# for some sets of input parameters, SVM will be slow to converge.  We will terminate early.
# This code will suppress warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)


pathtest='D:\cs474\lijj_files\cs474_hw3\HW3test.csv'
pathtrain='D:\cs474\lijj_files\cs474_hw3\HW3train.csv'
def read(path):
    with open(path,'r',encoding='UTF-8') as csvfile:
        reader=csv.reader(csvfile)
        rows=[row for row in reader]
        #header=row[0]
        data=np.array(rows)
        data=data.astype(np.float)
    return data
    #return header,data
    
testdata=read(pathtest)
traindata=read(pathtrain)
Xtrain=traindata[:,1:]
ytrain=traindata[:,0]
Xtest=testdata[:,1:]
ytest=testdata[:,0]

#a
#plt.title('HW3train')
#plt.scatter(traindata[:,1],traindata[:,2])
for x,y in zip(Xtrain, ytrain):
#     print(x1,x2,y)
    if y==1:
        col = 'blue'
    if y==2:
        col = 'red'
    if y==3:
        col = 'black'
    plt.scatter(x[0], x[1],  color=col)
    
plt.title('Scatterplot HW3Train')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('HW3trainscatter.png',dpi=300,bbox_inches='tight')
plt.show()

for x,y in zip(Xtest, ytest):
#     print(x1,x2,y)
    if y==1:
        col = 'blue'
    if y==2:
        col = 'red'
    if y==3:
        col = 'black'
    plt.scatter(x[0], x[1],  color=col)
    
plt.title('Scatterplot HW3Test')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('HW3testscatter.png',dpi=300,bbox_inches='tight')
plt.show()
#B. K-nn:
h = .03  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh 
x1_min, x1_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
x2_min, x2_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
x1mesh, x2mesh = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))

# Create color maps
cmap_light = ListedColormap(['lightblue', 'lightcoral', 'grey'])
cmap_bold = ListedColormap(['blue', 'red', 'black'])

#for n_neighbors in [3,20]:
for n_neighbors in [1,5,15]:
    # we create an instance of Neighbours Classifier and fit the data.
    
    clf = KNeighborsClassifier(n_neighbors, weights='uniform',algorithm='auto')
    clf.fit(Xtrain, ytrain)

    Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(x1mesh.shape)

    # Plot the training points with the mesh
'''
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytrain_colors = [y-1 for y in ytrain]
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Training Set' % (n_neighbors))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot the testing points with the mesh
    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y-1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Testing Set' % (n_neighbors))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    #Report training and testing accuracies
    print('Working on k=%i'%(n_neighbors))
    trainacc =  clf.score(Xtrain,ytrain) 
    testacc = clf.score(Xtest,ytest) 
    print('\tThe training accuracy is %.2f'%(trainacc))
    print('\tThe testing accuracy is %.2f'%(testacc))
'''

    # we create an instance of Neighbours Classifier and fit the data.
#b,5
# =============================================================================
# trainacc=np.zeros(30)
# testacc=np.zeros(30)
# klabel=np.linspace(1,30,30)
# for k in range (1,31):
#     # we create an instance of Neighbours Classifier and fit the data.    
#     clf = KNeighborsClassifier(k, weights='uniform',algorithm='auto')
#     clf=clf.fit(Xtrain, ytrain)
#     Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
#     # Put the result into a color plot
#     Z = Z.reshape(x1mesh.shape)
#     #ypred = clf.predict(Xtest)
#     trainacc[k-1] =  clf.score(Xtrain,ytrain) 
#     testacc[k-1] = clf.score(Xtest,ytest) 
# 
# plt.plot(klabel,trainacc)
# plt.xlim(0, 30)
# plt.ylim(0.7, 1)
# plt.title('Training accuracy as a function of k')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.show()
# 
# plt.plot(klabel,testacc)
# plt.xlim(0, 30)
# plt.ylim(0.7, 1)
# plt.title('Testing accuracy as a function of k')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.show()
# 
# print(trainacc)
# print(testacc)
#     
# =============================================================================
# =============================================================================
#b,6,7
# for n_neighbors in [21]:
#     # we create an instance of Neighbours Classifier and fit the data.
#     
#     clf = KNeighborsClassifier(n_neighbors, weights='uniform',algorithm='auto')
#     clf.fit(Xtrain, ytrain)
# 
#     Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
# 
#     # Put the result into a color plot
#     Z = Z.reshape(x1mesh.shape)
# 
#     # Plot the training points with the mesh
# 
#     plt.figure()
#     plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
#     ytrain_colors = [y-1 for y in ytrain]
#     plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
#     plt.xlim(x1_min, x1_max)
#     plt.ylim(x2_min, x2_max)
#     plt.title('%i-NN Training Set' % (n_neighbors))
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     
#     # Plot the testing points with the mesh
#     ypred = clf.predict(Xtest)
#     plt.figure()
#     plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
#     ytest_colors = [y-1 for y in ytest]
#     plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
#     plt.xlim(x1_min, x1_max)
#     plt.ylim(x2_min, x2_max)
#     plt.title('%i-NN Testing Set' % (n_neighbors))
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     
#     #Report training and testing accuracies
#     print('Working on k=%i'%(n_neighbors))
#     trainacc =  clf.score(Xtrain,ytrain) 
#     testacc = clf.score(Xtest,ytest) 
#     print('\tThe training accuracy is %.2f'%(trainacc))
#     print('\tThe testing accuracy is %.2f'%(testacc))
# 
# =============================================================================

# =============================================================================
# 
# #c1,2,3
# 
# # we create an instance of Neighbours Classifier and fit the data.
# 
# #clf = KNeighborsClassifier(n_neighbors, weights='uniform',algorithm='auto')
# clf=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,priors=None)
# clf.fit(Xtrain, ytrain)
# 
# Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
# 
# # Put the result into a color plot
# Z = Z.reshape(x1mesh.shape)
# 
# # Plot the training points with the mesh
# 
# plt.figure()
# plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
# ytrain_colors = [y-1 for y in ytrain]
# plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('Training Set--LDA')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# 
# # Plot the testing points with the mesh
# ypred = clf.predict(Xtest)
# plt.figure()
# plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
# ytest_colors = [y-1 for y in ytest]
# plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('Testing Set--LDA')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# 
# #Report training and testing accuracies
# #print('Working on k=%i'%(n_neighbors))
# trainacc =  clf.score(Xtrain,ytrain) 
# testacc = clf.score(Xtest,ytest) 
# print('\tThe training accuracy is %.2f'%(trainacc))
# print('\tThe testing accuracy is %.2f'%(testacc))
# 
#     
# =============================================================================
    
    
    

# =============================================================================
# #d1,2,3
# 
# # we create an instance of Neighbours Classifier and fit the data.
# 
# #clf = KNeighborsClassifier(n_neighbors, weights='uniform',algorithm='auto')
# #clf=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,priors=None)
# clf=QuadraticDiscriminantAnalysis(priors=None,reg_param=0.0)
# clf.fit(Xtrain, ytrain)
# 
# Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
# 
# # Put the result into a color plot
# Z = Z.reshape(x1mesh.shape)
# 
# # Plot the training points with the mesh
# 
# plt.figure()
# plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
# ytrain_colors = [y-1 for y in ytrain]
# plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('Training Set--QDA')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# 
# # Plot the testing points with the mesh
# ypred = clf.predict(Xtest)
# plt.figure()
# plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
# ytest_colors = [y-1 for y in ytest]
# plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('Testing Set--QDA')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# 
# #Report training and testing accuracies
# #print('Working on k=%i'%(n_neighbors))
# trainacc =  clf.score(Xtrain,ytrain) 
# testacc = clf.score(Xtest,ytest) 
# print('\tThe training accuracy is %.2f'%(trainacc))
# print('\tThe testing accuracy is %.2f'%(testacc))
# 
# =============================================================================
    
#E,SVM, ploynomial kernal

#clf = KNeighborsClassifier(n_neighbors, weights='uniform',algorithm='auto')
#clf=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,priors=None)
#clf=QuadraticDiscriminantAnalysis(priors=None,reg_param=0.0)
# =============================================================================
# max_iter=1000
# gamma=1.0
# Cvals=np.logspace(-4,2,25,base=10)
# trainacc=np.zeros(100).reshape(4,-1)
# testacc=np.zeros(100).reshape(4,-1)
# # =============================================================================
# for i in range (25):    
#      for p in [1,2,3,4]:
#           # we create an instance of Neighbours Classifier and fit the data.
#           clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
#           clf.fit(Xtrain, ytrain)
#       
#           Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
#       
#           # Put the result into a color plot
#           Z = Z.reshape(x1mesh.shape)
#       
#           # Plot the training points with the mesh
#  
#           plt.figure()
#           plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
#           ytrain_colors = [y-1 for y in ytrain]
#           plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
#           plt.xlim(x1_min, x1_max)
#           plt.ylim(x2_min, x2_max)
#           plt.title('%x degree Training Set  ' % (p))
#           plt.xlabel('Feature 1')
#           plt.ylabel('Feature 2')
#           
#           # Plot the testing points with the mesh
#           ypred = clf.predict(Xtest)
#           plt.figure()
#           plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
#           ytest_colors = [y-1 for y in ytest]
#           plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
#           plt.xlim(x1_min, x1_max)
#           plt.ylim(x2_min, x2_max)
#           plt.title('%x degree Testing Set ' % (p))
#           plt.xlabel('Feature 1')
#           plt.ylabel('Feature 2')
#  
#           
#           #Report training and testing accuracies
#           # print('Working on k=%i'%(n_neighbors))
#           trainacc[p-1,i] =clf.score(Xtrain,ytrain) 
#           testacc[p-1,i] = clf.score(Xtest,ytest) 
# print(trainacc)
# print(testacc)
# print(Cvals)
# =============================================================================
#      
# =============================================================================
# =============================================================================
# for i in [11]:    
#     for p in [2]:
#          # we create an instance of Neighbours Classifier and fit the data.
#          clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
#          clf.fit(Xtrain, ytrain)
#      
#          Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
#      
#          # Put the result into a color plot
#          Z = Z.reshape(x1mesh.shape)
#      
#          # Plot the training points with the mesh
# 
#          plt.figure()
#          plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
#          ytrain_colors = [y-1 for y in ytrain]
#          plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
#          plt.xlim(x1_min, x1_max)
#          plt.ylim(x2_min, x2_max)
#          plt.title('%x degree Training Set  ' % (p))
#          plt.xlabel('Feature 1')
#          plt.ylabel('Feature 2')
#          
#          # Plot the testing points with the mesh
#          ypred = clf.predict(Xtest)
#          plt.figure()
#          plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
#          ytest_colors = [y-1 for y in ytest]
#          plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
#          plt.xlim(x1_min, x1_max)
#          plt.ylim(x2_min, x2_max)
#          plt.title('%x degree Testing Set ' % (p))
#          plt.xlabel('Feature 1')
#          plt.ylabel('Feature 2')
# 
#          
#          #Report training and testing accuracies
#          # print('Working on k=%i'%(n_neighbors))
#          trainacc[p-1,i] =clf.score(Xtrain,ytrain) 
#          testacc[p-1,i] = clf.score(Xtest,ytest) 
# print(trainacc)
# print(testacc)
# print(Cvals)
#      

# =============================================================================

#G,SVM,ridical kernel
# =============================================================================
Cvals=np.logspace(-4,2,25,base=10)
gamma_vals=np.logspace(-2,2,25,base=10)
max_iter=1000
count=[0,0]
tmax=0
traintmax=0
# for c in (Cvals):    
#     for g in (gamma_vals):
#         
#         # we create an instance of Neighbours Classifier and fit the data.
#         #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
#         clf=SVC(C=c,kernel='rbf',gamma=g,shrinking=True,probability=False,max_iter=max_iter)
#         clf.fit(Xtrain, ytrain)
#      
#         Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
#         # Put the result into a color plot
#         Z = Z.reshape(x1mesh.shape)
#          
#         # Plot the training points with the mesh
#         #Report training and testing accuracies
#         # print('Working on k=%i'%(n_neighbors))
#         #trainacc[p-1,i] =clf.score(Xtrain,ytrain) 
#         #temp=[tmax,clf.score(Xtest,ytest)]
#         if (clf.score(Xtest,ytest)>tmax):
#             
#             tmax= clf.score(Xtest,ytest)
#             traintmax=clf.score(Xtrain,ytrain) 
#             count=[c,g]
# print (count)
# print (tmax)
# print(traintmax)
# =============================================================================

# we create an instance of Neighbours Classifier and fit the data.
#clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
# =============================================================================
# clf=SVC(C=0.1778279410038923,kernel='rbf',gamma=0.21544346900318834,shrinking=True,probability=False,max_iter=max_iter)
# clf.fit(Xtrain, ytrain)
# #      
# Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
# #      
# # Put the result into a color plot
# Z = Z.reshape(x1mesh.shape)
# #      
# # Plot the training points with the mesh
# # 
# plt.figure()
# plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
# ytrain_colors = [y-1 for y in ytrain]
# plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('Training Set')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# #          
# # Plot the testing points with the mesh
# ypred = clf.predict(Xtest)
# plt.figure()
# plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
# ytest_colors = [y-1 for y in ytest]
# plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('Testing Set ')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# =============================================================================
# 

#HW4 Q4

#max_iter=1000
#gamma=1.0
#Cvals=np.logspace(-4,2,25,base=10)

#4a
'''
depth=[1,2,3,4,10]
trainacc=np.zeros(5)
testacc=np.zeros(5)




#4b

depth=[1,2,3,4,5,6,7,8,9,10]
trainacc=np.zeros(10)
testacc=np.zeros(10)
# =============================================================================
#for i in range (5):   #4a
for i in range (10):     #4b   
          # we create an instance of Neighbours Classifier and fit the data.
          #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
          clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
          clf.fit(Xtrain, ytrain)
      
          Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
      
          # Put the result into a color plot
          Z = Z.reshape(x1mesh.shape)
      
          # Plot the training points with the mesh
          print (depth[i])
          plt.figure()
          plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
          ytrain_colors = [y-1 for y in ytrain]
          plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
          plt.xlim(x1_min, x1_max)
          plt.ylim(x2_min, x2_max)
          plt.title('%d depth Training Set  ' % (depth[i]))
          plt.xlabel('Feature 1')
          plt.ylabel('Feature 2')
          plt.show()
          # Plot the testing points with the mesh
          ypred = clf.predict(Xtest)
          plt.figure()
          plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
          ytest_colors = [y-1 for y in ytest]
          plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
          plt.xlim(x1_min, x1_max)
          plt.ylim(x2_min, x2_max)
          plt.title('%d depth Testing Set ' % (depth[i]))
          plt.xlabel('Feature 1')
          plt.ylabel('Feature 2')
          plt.show()
          #Report training and testing accuracies
          # print('Working on k=%i'%(n_neighbors))
          trainacc[i] =clf.score(Xtrain,ytrain) 
          testacc[i] = clf.score(Xtest,ytest) 
          
print(trainacc)
print(testacc)

plt.title('depth-accuracy training data')
plt.plot(depth,trainacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


plt.title('depth-accuracy test data')
plt.plot(depth,testacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()
'''



#4c

depth=[1,2,3,4,5,6,7,8,9,10]
numtreerange=[1,5,10,25,50,100,200]
trainacc=np.zeros(10)
testacc=np.zeros(10)
# =============================================================================
#for i in range (5):   #4a
for i in range (10):     #4b 
     tempscore=0
     temp_numtree=0
     for j in range(7):
    
          # we create an instance of Neighbours Classifier and fit the data.
          #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
          #clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
          clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[j],
                                     max_features=None,criterion='gini',max_depth=depth[i])  #bagging
          clf.fit(Xtrain, ytrain)
      
          Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
      
          # Put the result into a color plot
          Z = Z.reshape(x1mesh.shape)
      
          # Plot the training points with the mesh
          #if (tempscore<clf.score(Xtrain,ytrain)):
          #    tempscore=clf.score(Xtrain,ytrain)
          if (tempscore<clf.score(Xtest,ytest)):
              tempscore=clf.score(Xtest,ytest)
              temp_numtree=numtreerange[j]
              
              
     
     clf=RandomForestClassifier(bootstrap=True,n_estimators=temp_numtree,
                                     max_features=None,criterion='gini',max_depth=depth[i])  #bagging
     clf.fit(Xtrain, ytrain)
      
     Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
      
     # Put the result into a color plot
     Z = Z.reshape(x1mesh.shape)     
     print(temp_numtree)
     plt.figure()
     plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
     ytrain_colors = [y-1 for y in ytrain]
     plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
     plt.xlim(x1_min, x1_max)
     plt.ylim(x2_min, x2_max)
     plt.title('%d depth Training Set  ' % (depth[i]))
     plt.suptitle('%d tree num  ' % (temp_numtree))
     plt.xlabel('Feature 1')
     plt.ylabel('Feature 2')
     plt.show()
     # Plot the testing points with the mesh
     ypred = clf.predict(Xtest)
     plt.figure()
     plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
     ytest_colors = [y-1 for y in ytest]
     plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
     plt.xlim(x1_min, x1_max)
     plt.ylim(x2_min, x2_max)
     plt.title('%d depth Testing Set ' % (depth[i]))
     plt.suptitle('%d tree num  ' % (temp_numtree))
     plt.xlabel('Feature 1')
     plt.ylabel('Feature 2')
     plt.show()
     #Report training and testing accuracies
     # print('Working on k=%i'%(n_neighbors))
     trainacc[i] =clf.score(Xtrain,ytrain) 
     testacc[i] = clf.score(Xtest,ytest) 
          
print(trainacc)
print(testacc)

plt.title('depth-accuracy training data')
plt.plot(depth,trainacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


plt.title('depth-accuracy test data')
plt.plot(depth,testacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


'''
#4e
learnraterange=np.logspace(-3,0,15,base=10) #numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0
depth=[1,2,3,4,5,6,7,8,9,10]
numtreerange=[1,5,10,25,50,100,200]
trainacc=np.zeros(10)
testacc=np.zeros(10)
# =============================================================================
#for i in range (5):   #4a
for i in range (10):     #4b 
     tempscore=0
     temp_numtree=0
     temp_learningrate=0
     for j in range(7):
         for k in range(15):
    
              # we create an instance of Neighbours Classifier and fit the data.
              #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
              #clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
              #clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[j],max_features=None,criterion='gini',max_depth=depth[i])  #bagging
              clf=GradientBoostingClassifier(learning_rate=learnraterange[k],n_estimators=numtreerange[j],max_depth=depth[i])
              clf.fit(Xtrain, ytrain)
          
              Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
          
              # Put the result into a color plot
              Z = Z.reshape(x1mesh.shape)
          
              # Plot the training points with the mesh
              #use test data to get the coefficient
              
              #if (tempscore<clf.score(Xtest,ytest)):
              #    tempscore=clf.score(Xtest,ytest)
              if (tempscore<clf.score(Xtest,ytest)):
                  tempscore=clf.score(Xtest,ytest)
                  temp_numtree=numtreerange[j]
                  temp_learningrate=learnraterange[k]
              
              
     
     clf=GradientBoostingClassifier(learning_rate=temp_learningrate,n_estimators=temp_numtree,max_depth=depth[i])
     clf.fit(Xtrain, ytrain)
      
     Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])
      
     # Put the result into a color plot
     Z = Z.reshape(x1mesh.shape)     
     print( temp_numtree)
     print(temp_learningrate)
     plt.figure()
     plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
     ytrain_colors = [y-1 for y in ytrain]
     plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
     plt.xlim(x1_min, x1_max)
     plt.ylim(x2_min, x2_max)
     plt.title('%d depth Training Set  ' % (depth[i]))
     plt.suptitle('%d tree num  ' % (temp_numtree))
     plt.suptitle('%g learning rate  ' % (temp_learningrate))
     plt.xlabel('Feature 1')
     plt.ylabel('Feature 2')
     plt.show()
     # Plot the testing points with the mesh
     ypred = clf.predict(Xtest)
     plt.figure()
     plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
     ytest_colors = [y-1 for y in ytest]
     plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
     plt.xlim(x1_min, x1_max)
     plt.ylim(x2_min, x2_max)
     plt.title('%d depth Testing Set ' % (depth[i]))
     plt.suptitle('%d tree num  ' % (temp_numtree))
     plt.suptitle('%g learning rate  ' % (temp_learningrate))
     plt.xlabel('Feature 1')
     plt.ylabel('Feature 2')
     plt.show()
     #Report training and testing accuracies
     # print('Working on k=%i'%(n_neighbors))
     trainacc[i] =clf.score(Xtrain,ytrain) 
     testacc[i] = clf.score(Xtest,ytest) 
          
print(trainacc)
print(testacc)

plt.title('depth-accuracy training data')
plt.plot(depth,trainacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


plt.title('depth-accuracy test data')
plt.plot(depth,testacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()
'''
















