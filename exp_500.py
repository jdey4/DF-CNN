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
    #get_ipython().system('python ./main_train_cl.py --data_type CIFAR100_10 --data_percent 100 --model_type DFCNN --lifelong --save_mat_name CIFAR_res2.mat')
    get_ipython().system('python ./main_train_cl.py --data_type CIFAR100_10 --data_percent 100 --model_type PROG --lifelong --save_mat_name CIFAR_res2.mat')
    #!python ./main_train_cl.py --data_type CIFAR100_10 --data_percent 100 --model_type PROG --lifelong --save_mat_name CIFAR_res.mat


# In[ ]:


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


# In[ ]:


#saving the file
alg = ['Prog_NN', 'DF_CNN']
task_to_complete = 1
with open('./task_count.pickle','wb') as f:
    pickle.dump(task_to_complete, f)


# In[ ]:


filename = './slot_res'
tmp_file = './tmp'
data_folder = './Data/cifar-100-python'

if not os.path.exists(filename):
    os.mkdir(filename)

if not os.path.exists(tmp_file):
    os.mkdir(tmp_file)

if not os.path.exists(data_folder):
     os.mkdir('Data')
     os.mkdir(data_folder)

num_points_per_task = 500
total_task = 10
slot_fold = range(10)
shift_fold = range(2,3,1)
algs = range(2)
_cifar100_task_labels_10 = [[0,1,2,3,4,5,6,7,8,9],
                            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]


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

        for t in range(total_task):
            tmp = [_cifar100_task_labels_10[t]]
            tmp.extend(_cifar100_task_labels_10[0:9])
            
            with open('./task_labels.pickle','wb') as f:
                pickle.dump(tmp, f)

            get_ipython().system('rm ./Data/cifar100_mtl_data_group_410_80_1000_10.pkl')
            experiment()
            res = unpickle('./tmp/res.pickle')
            with open(filename+'/'+alg[0]+str(shift)+'_'+str(slot)+'_'+str(t+1)+'.pickle','wb') as f:
                    pickle.dump(res,f)
                
                
get_ipython().system('sudo shutdown now')

