import numpy as np
import cv2
import os
import random
import pandas as pd

class ImageDataLoader():
    def __init__(self, data_path, gt_path,split, shuffle=False, gt_downsample=False, pre_load=False, Dataset="fdst"):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.split = split
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename))]
        self.data_files.sort()
        print(self.data_files)
        self.shuffle = shuffle
        self.Dataset=Dataset
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.num_samples)
        if self.pre_load:
            print ('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                #print(fname)
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                if Dataset=="fdst":
                  ht_1 = (540/4)*4  
                  wd_1 = (960/4)*4 
                else:
                  ht_1 = (ht/4)*4  
                  wd_1 = (wd/4)*4
                #print(ht_1)
                img = cv2.resize(img,(int(wd_1),int(ht_1)))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                if self.split == 'train_split':
                    den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).to_numpy()                        
                    den  = den.astype(np.float32, copy=False)

                    if self.gt_downsample:
                        wd_2 = wd_1/4
                        ht_2 = ht_1/4
                        den = cv2.resize(den,(int(wd_2),int(ht_2)))                
                        den = den * ((wd_1*ht_1)/(wd_2*ht_2))
                    else:
                        den = cv2.resize(den,(wd_1,ht_1))
                        den = den * ((wd*ht)/(wd_1*ht_1))
                        
                    den = den.reshape((1,1,den.shape[0],den.shape[1]))
                else:
                    den = []        
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 100 == 0:                    
                    print ('Loaded ', idx, '/', self.num_samples, 'files')
               
            print ('Completed Loading ', idx, 'files')
        
        
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(self.id_list)        
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list
       
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:                    
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                if Dataset=="fdst":
                  ht_1 = (540/4)*4  
                  wd_1 = (960/4)*4 
                else:
                  ht_1 = (ht/4)*4  
                  wd_1 = (wd/4)*4

                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                if self.split == 'train_split':
                    den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                    den  = den.astype(np.float32, copy=False)
                    if self.gt_downsample:
                        wd_2 = wd_1/4
                        ht_2 = ht_1/4
                        den = cv2.resize(den,(int(wd_2),int(ht_2)))                
                        den = den * ((wd_1*ht_1)/(wd_2*ht_2))
                    else:
                        den = cv2.resize(den,(wd_1,ht_1))
                        den = den * ((wd*ht)/(wd_1*ht_1))
                        
                    den = den.reshape((1,1,den.shape[0],den.shape[1]))
                else:
                    den = []           
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                


# ImageDataLoader_Val_Test    
        
class ImageDataLoader_Val_Test():
    def __init__(self, data_path, gt_path,split,start_frame,end_frame,shuffle=False, gt_downsample=False, pre_load=False, Dataset="fdst"):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        print('working')
        self.data_path = data_path
        self.gt_path = gt_path
        self.split = split
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.Dataset= Dataset
        
        # self.data_files = [filename for filename in os.listdir(data_path) \
        #                    if os.path.isfile(os.path.join(data_path,filename))]
        self.data_files = []
        for i in range(self.start_frame,self.end_frame+1):

            if Dataset=="fdst":
              self.data_files.append('{:03d}.jpg'.format(i))
            else:
              self.data_files.append('{:04d}.jpg'.format(i))

        
        
        self.data_files.sort()
        print(self.data_files)
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.num_samples)
        if self.pre_load:
            print ('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                #print(fname)
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                if Dataset=="fdst":
                  ht_1 = (540/4)*4  
                  wd_1 = (960/4)*4 
                else:
                  ht_1 = (ht/4)*4  
                  wd_1 = (wd/4)*4

                img = cv2.resize(img,(int(wd_1),int(ht_1)))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                if self.split == 'train_split':
                    den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).to_numpy()                        
                    den  = den.astype(np.float32, copy=False)

                    if self.gt_downsample:
                        wd_2 = wd_1/4
                        ht_2 = ht_1/4
                        den = cv2.resize(den,(int(wd_2),int(ht_2)))                
                        den = den * ((wd_1*ht_1)/(wd_2*ht_2))
                    else:
                        den = cv2.resize(den,(wd_1,ht_1))
                        den = den * ((wd*ht)/(wd_1*ht_1))
                        
                    den = den.reshape((1,1,den.shape[0],den.shape[1]))
                else:
                    den = []        
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 100 == 0:                    
                    print ('Loaded ', idx, '/', self.num_samples, 'files')
               
            print ('Completed Loading ', idx, 'files')
        
        
    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(self.id_list)        
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list
       
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:                    
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                if Dataset=="fdst":
                  ht_1 = (540/4)*4  
                  wd_1 = (960/4)*4 
                else:
                  ht_1 = (ht/4)*4  
                  wd_1 = (wd/4)*4

                img = cv2.resize(img,(wd_1,ht_1))
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                if self.split == 'train_split':
                    den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                    den  = den.astype(np.float32, copy=False)
                    if self.gt_downsample:
                        wd_2 = wd_1/4
                        ht_2 = ht_1/4
                        den = cv2.resize(den,(wd_2,ht_2))                
                        den = den * ((wd_1*ht_1)/(wd_2*ht_2))
                    else:
                        den = cv2.resize(den,(wd_1,ht_1))
                        den = den * ((wd*ht)/(wd_1*ht_1))
                        
                    den = den.reshape((1,1,den.shape[0],den.shape[1]))
                else:
                    den = []           
                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
