#%%
from itertools import product
import pandas as pd
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#%%
algs = ["Prog_NN", "DF_CNN"]
slot_fold = 10
shift_fold = 6
total_tasks = 10
epoch_indx = list(range(200,2200,200))

for alg in algs:
    for shift in range(shift_fold):
        for slot in range(slot_fold):
            df = pd.DataFrame()
            shifts = []
            base_tasks=[]
            accuracies_across_tasks = []

            # find multitask accuracies
            filename = '/Users/jayantadey/DF-CNN/slot_res/'+alg+str(shift+1)+'_'+str(slot)+'.pickle'
            accuracy = np.asarray(unpickle(filename))
            accuracy = accuracy[epoch_indx,:]
            for base_task in range(total_tasks):
                shifts.append(shift+1)
                base_tasks.append(base_task+1)
                accuracies_across_tasks.append(accuracy[base_task,0])

            df['data_fold'] = shifts
            df['task'] = base_tasks
            df['task_1_accuracy'] = accuracies_across_tasks

            file_to_save = 'reformed_res/'+alg+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            with open(file_to_save, 'wb') as f:
                pickle.dump(df, f)
# %%
