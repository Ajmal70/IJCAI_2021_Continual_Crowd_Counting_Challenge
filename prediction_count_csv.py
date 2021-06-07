# this code save the prediction counts from 1900 csv files to a single csv file for evaluation purpose.


import numpy as np
import glob
import csv

pred_path= "/pred/*.csv"

pred_list = []
for pred in glob.glob(pred_path):
        pred_data = np.genfromtxt(pred, delimiter=',')     
        pred_count = np.sum(pred_data)
        print ('prediction_count: %0.f' %(pred_count))
        pred_list.append(pred_count)

np.savetxt("pred_val_supervised.csv", pred_list, delimiter=",", fmt="%0.f")
