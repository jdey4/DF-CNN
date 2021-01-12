#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import numpy as np
from itertools import product
import os
import tensorflow as tf
import tensorflow.keras as keras
import warnings
warnings.filterwarnings("default", category=DeprecationWarning)


# In[ ]:


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def change_label(label, task):
    labels = label.copy()
    lbls_to_change = range(0,10,1)
    lbls_to_transform = range((task-1)*10,task*10,1)

    for count, i in enumerate(lbls_to_change):
        indx = np.where(labels == i)
    
        labels[indx] = -lbls_to_transform[count]
    
    for count, i in enumerate(lbls_to_transform):
        indx = np.where(labels == i)
    
        labels[indx] = lbls_to_change[count]
    
    indx = np.where(labels<0)
    labels[indx] = -labels[indx]
    
    return labels


# In[ ]:


def cross_val_data(data_x, data_y, num_points_per_task, slot_no, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    sample_per_class = num_points_per_task//total_task

    for task in range(total_task):
        for class_no in range(task*10,(task+1)*10,1):
            indx = np.roll(idx[class_no],(shift-1)*100)
            
            if class_no==0 and task==0:
                train_x = x[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class],:]
                train_y = y[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class]]
            else:
                train_x = np.concatenate((train_x, x[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class],:]), axis=0)
                train_y = np.concatenate((train_y, y[indx[slot_no*sample_per_class:(slot_no+1)*sample_per_class]]), axis=0)
                
            if class_no==0:
                test_x = x[indx[500:600],:]
                test_y = y[indx[500:600]]
            else:
                test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
                test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)     
            
    return train_x, train_y, test_x, test_y


# In[ ]:


def experiment():
    get_ipython().system('python ./main_train_cl.py --data_type CIFAR100_10 --data_percent 100 --model_type DFCNN --lifelong --save_mat_name CIFAR_res2.mat')
    #get_ipython().system('python ./main_train_cl.py --data_type CIFAR100_10 --data_percent 100 --model_type PROG --lifelong --save_mat_name CIFAR_res2.mat')
    #!python ./main_train_cl.py --data_type CIFAR100_10 --data_percent 100 --model_type PROG --lifelong --save_mat_name CIFAR_res.mat


# In[ ]:


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


# In[ ]:


#saving the file
alg = ['Prog_NN', 'DF_CNN']
task_to_complete = 10
with open('./task_count.pickle','wb') as f:
    pickle.dump(task_to_complete, f)


# In[ ]:


filename = './slot_res'
time_res = './time_res'
mem_res = './mem_res'
tmp_file = './tmp'
data_folder = './Data/cifar-100-python'

if not os.path.exists(filename):
    os.mkdir(filename)

if not os.path.exists(time_res):
    os.mkdir(time_res)

if not os.path.exists(mem_res):
    os.mkdir(mem_res)

if not os.path.exists(tmp_file):
    os.mkdir(tmp_file)

if not os.path.exists(data_folder):
     os.mkdir('Data')
     os.mkdir(data_folder)

num_points_per_task = 500
slot_fold = range(1,10,1)
shift_fold = range(1,2,1)
algs = range(2)

for shift in shift_fold:
    for slot in slot_fold:
        tmp_train = {}
        tmp_test = {}
        train_x, train_y, test_x, test_y = cross_val_data(
            data_x,data_y,num_points_per_task,slot,shift=shift
        )
        tmp_train[b'data'] = train_x
        tmp_train[b'fine_labels'] = train_y
        tmp_test[b'data'] = test_x
        tmp_test[b'fine_labels'] = test_y
        
        with open('./Data/cifar-100-python/train.pickle', 'wb') as f:
            pickle.dump(tmp_train, f)
            
        with open('./Data/cifar-100-python/test.pickle', 'wb') as f:
            pickle.dump(tmp_test, f)
    
        get_ipython().system('rm ./Data/cifar100_mtl_data_group_410_80_1000_10.pkl')
        experiment()
        res = unpickle('./tmp/res.pickle')
        with open(filename+'/'+alg[1]+str(shift)+'_'+str(slot)+'.pickle','wb') as f:
                pickle.dump(res,f)
                
        res = unpickle('./time_info.pickle')
        with open(time_res+'/'+alg[1]+str(shift)+'_'+str(slot)+'.pickle','wb') as f:
                pickle.dump(res,f)

        res = unpickle('./memory_info.pickle')
        with open(mem_res+'/'+alg[1]+str(shift)+'_'+str(slot)+'.pickle','wb') as f:
                pickle.dump(res,f)
                
#get_ipython().system('sudo shutdown now')
